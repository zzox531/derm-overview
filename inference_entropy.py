"""
run_inference.py — CLIP-based dermatology inference script.

Usage:
    python run_inference.py [--tasks caption zeroshot] [--budget-caption 1024] [--budget-zeroshot 256]

Outputs are saved under:
    results/
    ├── caption/
    │   ├── <model_name>/
    │   │   └── <image_stem>.<ext>   (visualisation / saliency maps, if any)
    │   └── ...
    ├── zeroshot/
    │   ├── <model_name>/
    │   │   └── <image_stem>.<ext>
    │   └── ...
    └── logs/
        └── run_<timestamp>.log
"""

import argparse
import gc
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import subprocess

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger(__name__)


def cuda_check(logger: logging.Logger) -> torch.device:
    """Print a detailed CUDA / device report and return the best available device."""
    logger.info("=" * 60)
    logger.info("DEVICE CHECK")
    logger.info("=" * 60)
    logger.info(f"PyTorch version : {torch.__version__}")

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        logger.info(f"CUDA available  : YES  ({n} device{'s' if n > 1 else ''})")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024 ** 3
            logger.info(
                f"  GPU {i}: {props.name}  |  "
                f"Compute {props.major}.{props.minor}  |  "
                f"VRAM {mem_gb:.1f} GB"
            )
        device = torch.device("cuda")
    else:
        logger.warning("CUDA available  : NO  — running on CPU (inference will be slow)")
        device = torch.device("cpu")

    logger.info(f"Active device   : {device}")
    logger.info("=" * 60)
    return device


def log_gpu_memory(logger: logging.Logger, label: str = "") -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
        logger.info(
            f"  GPU memory {label}: "
            f"{allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
        )


def make_output_dirs(base: Path, model_name: str, tasks: list[str]) -> dict[str, Path]:
    """Create and return per-task/per-model output directories."""
    safe_name = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    dirs = {}
    for task in tasks:
        d = base / task / safe_name
        d.mkdir(parents=True, exist_ok=True)
        dirs[task] = d
    return dirs


def release_results(results: list) -> None:
    """Move any tensors in explain() results to CPU and drop GPU references."""
    for item in results:
        iv = item.get("interaction_values")
        if iv is not None and hasattr(iv, "cpu"):
            item["interaction_values"] = iv.cpu()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = [
    # {
    #     "name": "coca_ViT-B-32",
    #     "backend": "open_clip",
    #     "pretrained": "laion2b_s13b_b90k",
    #     "budget_caption": 1024,
    #     "budget_zeroshot": 256,
    # },
    {
        "name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "backend": "open_clip",
        "hf_tokenizer_name": (
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        ),
        # BiomedCLIP has 196 image patches (14×14) vs CoCa's 49 (7×7),
        # so the coalition space is ~4× larger — use a smaller budget.
        "budget_caption": 256,
        "budget_zeroshot": 64,
    },
]

DERM1M_ENTRIES = [
    # {"filename": "pubmed/0d_59_PMC4458964_IJD_60_321e_g003_0.png", "index": 132556},
    # {"filename": "IIYI/2281_1.png", "index": 126},
]

HAM7 = [
    "melanocytic nevus",
    "melanoma",
    "basal cell carcinoma",
    "benign keratosis",
    "actinic keratosis",
    "vascular lesion",
    "dermatofibroma",
]

HAM_IMAGES = [
    "ham_images/sample_2_melanoma.jpg",
    # "ham_images/sample_3_melanoma.jpg",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CLIP-based dermatology inference (caption + zero-shot)."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["caption", "zeroshot"],
        default=["caption", "zeroshot"],
        help="Which tasks to run (default: both).",
    )
    parser.add_argument(
        "--budget-caption",
        type=int,
        default=None,
        help="Override token/patch budget for caption task for ALL models.",
    )
    parser.add_argument(
        "--budget-zeroshot",
        type=int,
        default=None,
        help="Override token/patch budget for zero-shot task for ALL models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Root directory for all outputs (default: ./results).",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="derm_train_images",
        help="Root directory for Derm1M images (default: derm_train_images).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Logging ---
    logger = setup_logging(args.output_dir / "logs")
    logger.info("Starting inference run")
    logger.info(f"Tasks        : {args.tasks}")
    logger.info(f"Output dir   : {args.output_dir.resolve()}")

    # --- CUDA check ---
    device = cuda_check(logger)

    # --- Lazy imports (keep startup fast when just checking --help) ---
    try:
        from huggingface_hub import login
        from datasets import load_dataset

        from src.model_loader import load_model
        from src.tasks import CaptionTask, ZeroShotTask
        from src.datasets_adapter import from_derm1m, from_image_folder
        from src.runner import explain
    except ImportError as exc:
        logger.error(f"Import failed: {exc}")
        logger.error("Make sure you have installed all dependencies and src/ is on PYTHONPATH.")
        sys.exit(1)

    # --- HuggingFace login ---
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN environment variable is not set; private datasets may fail.")
    else:
        login(token=hf_token)
        logger.info("Logged in to HuggingFace Hub.")

    # --- Build tasks ---
    caption_task = None
    zeroshot_task = None

    if "caption" in args.tasks:
        logger.info("Loading Derm1M dataset for caption task …")
        ds = load_dataset("redlessone/Derm1M")
        caption_samples = from_derm1m(ds, DERM1M_ENTRIES, image_root=args.image_root)
        caption_task = CaptionTask(samples=caption_samples)
        logger.info(f"Caption task ready — {len(caption_samples)} sample(s).")

    if "zeroshot" in args.tasks:
        zs_samples = from_image_folder(HAM_IMAGES)
        zeroshot_task = ZeroShotTask(
            samples=zs_samples,
            class_names=HAM7,
            prompt_template=lambda c: f"This image shows a case of {c}",
            explain_classes="top1",
            top_k=3,
        )
        logger.info(f"Zero-shot task ready — {len(zs_samples)} sample(s), {len(HAM7)} classes.")

    # --- Run models ---
    all_results: dict = {}

    for cfg in MODELS:
        # Separate budget fields from model-loader kwargs
        model_name       = cfg["name"]
        budget_caption   = args.budget_caption  or cfg.get("budget_caption",  2 ** 10)
        budget_zeroshot  = args.budget_zeroshot or cfg.get("budget_zeroshot", 2 ** 8)
        loader_kwargs    = {k: v for k, v in cfg.items()
                            if k not in ("budget_caption", "budget_zeroshot")}

        logger.info(f"\n{'─' * 60}")
        logger.info(f"Loading model: {model_name}")
        log_gpu_memory(logger, "before load")

        model = load_model(**loader_kwargs)
        out_dirs = make_output_dirs(args.output_dir, model_name, args.tasks)
        all_results[model_name] = {}

        if caption_task is not None:
            logger.info(f"  Running caption task (budget={budget_caption}) …")
            caption_results = explain(
                model,
                caption_task,
                budget=budget_caption,
                output_dir=out_dirs["caption"],
            )
            release_results(caption_results)
            all_results[model_name]["caption"] = caption_results
            logger.info(f"  Caption results saved → {out_dirs['caption']}")

        if zeroshot_task is not None:
            logger.info(f"  Running zero-shot task (budget={budget_zeroshot}) …")
            zs_results = explain(
                model,
                zeroshot_task,
                budget=budget_zeroshot,
                output_dir=out_dirs["zeroshot"],
            )
            release_results(zs_results)
            all_results[model_name]["zeroshot"] = zs_results
            logger.info(f"  Zero-shot results saved → {out_dirs['zeroshot']}")

        # --- Thorough GPU cleanup before loading the next model ---
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        log_gpu_memory(logger, "after cleanup")

    logger.info("\nAll models done. Results summary:")
    for model_name, tasks in all_results.items():
        for task_name, result in tasks.items():
            logger.info(f"  [{model_name}]  {task_name}: {result}")

    logger.info("Inference run complete.")


if __name__ == "__main__":
    main()