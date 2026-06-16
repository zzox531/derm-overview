"""
run_inference.py — CLIP-based dermatology inference script.

Usage:
    python run_inference.py [--tasks caption zeroshot ins_del] \
                            [--budget-caption 1024] \
                            [--budget-zeroshot 256] \
                            [--budget-ins-del 1024] \
                            [--ins-del-batch-size 64] \
                            [--ins-del-p 0.5]

Outputs are saved under:
    results/
    ├── caption/
    │   └── <model_name>/   (FIxLIP saliency maps)
    ├── zeroshot/
    │   └── <model_name>/   (FIxLIP saliency maps)
    ├── ins_del/
    │   └── <model_name>/   (*_curves.png, *_curves.csv, *_attr.npy)
    └── logs/
        └── run_<timestamp>.log

Insertion / deletion behaviour
-------------------------------
* "ins_del" can be requested as a standalone task OR combined with
  "caption" / "zeroshot".

  Standalone  (--tasks ins_del):
    AttributionValues are estimated via Monte Carlo (budget = --budget-ins-del).
    Runs on BOTH HAM_IMAGES (zero-shot prompts) and DERM1M_ENTRIES (captions).

  Combined  (--tasks caption ins_del  OR  --tasks zeroshot ins_del):
    FIxLIP interaction values from the caption / zeroshot run are forwarded
    directly to the ins/del sweep — no extra attribution budget is spent.
    This is the recommended mode.
"""

import argparse
import gc
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
# from ibm_db import columns
import pandas as pd
# from pyparsing import col
import torch

# ---------------------------------------------------------------------------
# Helpers (unchanged from original)
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"run_{timestamp}.log"
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
    logger.info("=" * 60)
    logger.info("DEVICE CHECK")
    logger.info("=" * 60)
    logger.info(f"PyTorch version : {torch.__version__}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        logger.info(f"CUDA available  : YES  ({n} device{'s' if n > 1 else ''})")
        for i in range(n):
            props   = torch.cuda.get_device_properties(i)
            mem_gb  = props.total_memory / 1024 ** 3
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
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
        logger.info(
            f"  GPU memory {label}: "
            f"{allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
        )


def make_output_dirs(base: Path, model_name: str, tasks: list[str]) -> dict[str, Path]:
    safe = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    dirs = {}
    for task in tasks:
        d = base / task / safe
        d.mkdir(parents=True, exist_ok=True)
        dirs[task] = d
    return dirs


def release_results(results: list) -> None:
    for item in results:
        iv = item.get("interaction_values")
        if iv is not None and hasattr(iv, "cpu"):
            item["interaction_values"] = iv.cpu()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = [
    # ── General VLMs ──────────────────────────────────────────────
    {
        "name": "ViT-B-16",
        "backend": "open_clip",
        "pretrained": "openai",
        "budget_caption": 2**19,
        "budget_zeroshot": 2**19,
    },
    # {
    #     "name": "ViT-B-16-SigLIP-256",
    #     "backend": "open_clip",
    #     "hf_tokenizer_name": "timm/ViT-B-16-SigLIP-256",
    #     "hf_tokenizer_max_length": 64,
    #     "pretrained": "webli",
    #     "budget_caption": 1024,
    #     "budget_zeroshot": 256,
    # },
    # {
    #     "name": "coca_ViT-B-32",
    #     "backend": "open_clip",
    #     "pretrained": "laion2b_s13b_b90k",
    #     "budget_caption": 1024,
    #     "budget_zeroshot": 256,
    # },

    # # ── Biomedical VLMs ───────────────────────────────────────────
    # {
    #     "name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    #     "backend": "open_clip",
    #     "hf_tokenizer_name": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    #     "budget_caption": 256,
    #     "budget_zeroshot": 64,
    # },
    # {
    #     "name": "hf-hub:redlessone/DermLIP_ViT-B-16",
    #     "backend": "open_clip",
    #     "budget_caption": 1024,
    #     "budget_zeroshot": 256,
    # },
    # {
    #     "name": "suinleelab/monet",
    #     "backend": "huggingface",
    #     "budget_zeroshot": 64,
    # },
]

DERM1M_ENTRIES = [
    # {"filename": "pubmed/0d_59_PMC4458964_IJD_60_321e_g003_0.png", "index": 132556},
    # {"filename": "pubmed/d5_e8_PMC3291114_DRP2012_925023.004.png", "index": 37},
    # {"filename": "pubmed/00_39_PMC10381143_jimaging-09-00148-g011_1.jpg", "index": 151106},
    # {"filename": "youtube/XwA-mCenqm4_frame_481_0.jpg", "index": 141},
    # {"filename": "youtube/5Fe4aaMKiWE_frame_1051_0_0.jpg", "index": 1356},
    # {"filename": "youtube/Y5AL8FruOJE_frame_12081_0_0.jpg", "index": 1698},
    # {"filename": "IIYI/27732_1.png", "index": 3846},
    # {"filename": "IIYI/18845_2.png", "index": 8098},
    # {"filename": "IIYI/20344_2.png", "index": 21523},
]

HAM7 = [
    "black dog next to a yellow hydrant"
    # "melanocytic nevus",
    # "melanoma",
    # "basal cell carcinoma",
    # "benign keratosis",
    # "actinic keratosis",
    # "vascular lesion",
    # "dermatofibroma",
]

HAM_IMAGES = [
    "sanity_check/dog_and_hydrant.jpg"
    # "ham_images/sample_0_melanocytic_Nevi.jpg",
    # "ham_images/sample_1_melanocytic_Nevi.jpg",
    # "ham_images/sample_2_melanoma.jpg",
    # "ham_images/sample_3_melanoma.jpg",
    # "ham_images/sample_4_melanocytic_Nevi.jpg",
]

# PAD_ROOT = "pad_images"
PAD_LABELS = [
    # "melanoma",
    # "basal cell carcinoma",
    # "actinic keratosis",
    # "benign keratosis",
    # "vascular lesion",
    # "dermatofibroma",
]

PAD_IMAGES = [
    # "pad_images/PAT_53_82_657.png"
]

PAD_TARGETS = []

# DAFFODIL_ROOT = "daffodil_images"
DAFFODIL_LABELS = [
    # "acne",
    # "hyperpigmentation",
    # "nail psoriasis",
    # "sjs ten",
    # "vitiligo",
]

DAFFODIL_IMAGES = []
DAFFODIL_TARGETS = []

def download_pad_daffodil ():
    global PAD_IMAGES, PAD_TARGETS, DAFFODIL_IMAGES, DAFFODIL_TARGETS
    dirs = ["pad_images", "daffodil_images"]
    
    df = pd.read_csv(os.path.join(dirs[0], "meta.csv"))
    pad_imgs = df['filename'].tolist()
    PAD_TARGETS = df['label'].tolist()
    
    df = pd.read_csv(os.path.join(dirs[1], "meta.csv"))
    daffodil_imgs = df['filename'].tolist()
    DAFFODIL_TARGETS = df['label'].tolist()
    
    PAD_IMAGES = [os.path.join(dirs[0], f) for f in pad_imgs]
    DAFFODIL_IMAGES = [os.path.join(dirs[1], f) for f in daffodil_imgs]
    
    print(PAD_IMAGES)
    print(PAD_TARGETS)
    print(DAFFODIL_IMAGES)
    print(DAFFODIL_TARGETS)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CLIP-based dermatology inference (caption + zero-shot + ins/del)."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["caption", "zeroshot", "ins_del"],
        default=["caption", "zeroshot"],
        help=(
            "Which tasks to run (default: caption zeroshot).  "
            "Add 'ins_del' to also compute insertion/deletion curves.  "
            "Combined modes (e.g. 'caption ins_del') reuse FIxLIP values; "
            "standalone 'ins_del' estimates attributions via Monte Carlo."
        ),
    )
    parser.add_argument("--budget-caption",   type=int, default=None,
                        help="Override FIxLIP budget for caption task.")
    parser.add_argument("--budget-zeroshot",  type=int, default=None,
                        help="Override FIxLIP budget for zero-shot task.")
    parser.add_argument("--budget-ins-del",   type=int, default=1024,
                        help="Attribution budget for standalone ins/del (default: 1024).")
    parser.add_argument("--ins-del-batch-size", type=int, default=64,
                        help="Batch size for masked-image scoring in ins/del sweep.")
    parser.add_argument("--ins-del-p",         type=float, default=0.5,
                        help="Banzhaf masking probability for standalone ins/del (default: 0.5).")
    parser.add_argument("--output-dir",  type=Path, default=Path("results"),
                        help="Root directory for all outputs (default: ./results).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    tasks  = args.tasks
    do_ins_del = "ins_del" in tasks

    # --- Logging ---
    logger = setup_logging(args.output_dir / "logs")
    logger.info("Starting inference run")
    logger.info(f"Tasks        : {tasks}")
    logger.info(f"Output dir   : {args.output_dir.resolve()}")

    # --- CUDA check ---
    device = cuda_check(logger)  # noqa: F841 (used inside loaded model)

    # --- Lazy imports ---
    try:
        from huggingface_hub import login
        from datasets import load_dataset

        from src.model_loader import load_model
        from src.tasks import CaptionTask, ZeroShotTask
        from src.datasets_adapter import from_derm1m, from_image_folder
        from src.runner import explain
    except ImportError as exc:
        logger.error(f"Import failed: {exc}")
        logger.error("Make sure dependencies are installed and src/ is on PYTHONPATH.")
        sys.exit(1)

    if do_ins_del:
        from src.ins_del import InsDelTask  # noqa: F401 (used below)

    # --- HuggingFace login ---
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set; private datasets may fail.")
    else:
        login(token=hf_token)
        logger.info("Logged in to HuggingFace Hub.")

    # --- Build tasks ---
    download_pad_daffodil()
    
    run_caption  = "caption"  in tasks
    run_zeroshot = "zeroshot" in tasks
    run_ins_del_standalone = do_ins_del and not run_caption and not run_zeroshot

    caption_task          = None
    zeroshot_task         = None
    ins_del_tasks: list   = []  # may hold one or two standalone ins/del tasks

    # Caption samples (Derm1M) — loaded if caption OR standalone ins/del is needed
    # and DERM1M_ENTRIES is non-empty.
    caption_samples = None
    if (run_caption or run_ins_del_standalone) and len(DERM1M_ENTRIES) > 0:
        logger.info("Loading Derm1M dataset")
        ds = load_dataset("redlessone/Derm1M")
        caption_samples = from_derm1m(ds, DERM1M_ENTRIES)

    # Zero-shot samples — loaded if zeroshot OR standalone ins/del is needed.
    ham_samples = None
    pad_samples = None
    daffodil_samples = None
    if (run_zeroshot or run_ins_del_standalone):
        if len(HAM_IMAGES) > 0:
            ham_samples = from_image_folder(HAM_IMAGES)
        if len(PAD_IMAGES) > 0:
            pad_samples = from_image_folder(PAD_IMAGES)
        if len(DAFFODIL_IMAGES) > 0:
            daffodil_samples = from_image_folder(DAFFODIL_IMAGES)

    if run_caption:
        if caption_samples is None or len(caption_samples) == 0:
            logger.warning("Caption task requested but DERM1M_ENTRIES is empty — skipping.")
        else:
            caption_task = CaptionTask(samples=caption_samples)
            logger.info(f"Caption task ready — {len(caption_samples)} sample(s).")

    zeroshot_tasks: list[tuple[str, ZeroShotTask]] = []
    if run_zeroshot:
        if ham_samples is None or len(ham_samples) == 0:
            logger.warning("Zero-shot task requested but HAM_IMAGES is empty — skipping HAM zero-shot.")
        else:
            zeroshot_tasks.append(
                (
                    "HAM",
                    ZeroShotTask(
                        samples=ham_samples,
                        class_names=HAM7,
                        prompt_template=lambda c: f"{c}",
                        explain_classes="top1",
                        top_k=3,
                    ),
                )
            )
            logger.info(
                f"HAM zero-shot task ready — {len(ham_samples)} sample(s), {len(HAM7)} classes."
            )

        if pad_samples is None or len(pad_samples) == 0:
            logger.warning("PAD zero-shot not loaded or empty — skipping PAD dataset.")
        else:
            zeroshot_tasks.append(
                (
                    "PAD",
                    ZeroShotTask(
                        samples=pad_samples,
                        class_names=PAD_LABELS,
                        prompt_template=lambda c: f"This image shows a case of {c}",
                        explain_classes="top1",
                        top_k=3,
                    ),
                )
            )
            logger.info(
                f"PAD zero-shot task ready — {len(pad_samples)} sample(s), {len(PAD_LABELS)} classes."
            )

        if daffodil_samples is None or len(daffodil_samples) == 0:
            logger.warning("Daffodil zero-shot not loaded or empty — skipping Daffodil dataset.")
        else:
            zeroshot_tasks.append(
                (
                    "Daffodil",
                    ZeroShotTask(
                        samples=daffodil_samples,
                        class_names=DAFFODIL_LABELS,
                        prompt_template=lambda c: f"This image shows a case of {c}",
                        explain_classes="top1",
                        top_k=3,
                    ),
                )
            )
            logger.info(
                f"Daffodil zero-shot task ready — {len(daffodil_samples)} sample(s), {len(DAFFODIL_LABELS)} classes."
            )

    if run_ins_del_standalone:
        # Standalone ins/del runs over HAM/PAD/Daffodil zero-shot prompts
        # and DERM1M_ENTRIES (captions), depending on which are populated.
        if ham_samples is not None and len(ham_samples) > 0:
            ins_del_tasks.append(
                InsDelTask(
                    samples=ham_samples,
                    class_names=HAM7,
                    prompt_template=lambda c: f"{c}",
                    explain_classes="top1",
                    budget=args.budget_ins_del,
                    p=args.ins_del_p,
                )
            )
            logger.info(
                f"Standalone ins/del (HAM/zero-shot) ready — {len(ham_samples)} sample(s).  "
                f"MC budget={args.budget_ins_del}, p={args.ins_del_p}"
            )

        if pad_samples is not None and len(pad_samples) > 0:
            ins_del_tasks.append(
                InsDelTask(
                    samples=pad_samples,
                    class_names=PAD_LABELS,
                    prompt_template=lambda c: f"This image shows a case of {c}",
                    explain_classes="top1",
                    budget=args.budget_ins_del,
                    p=args.ins_del_p,
                )
            )
            logger.info(
                f"Standalone ins/del (PAD/zero-shot) ready — {len(pad_samples)} sample(s).  "
                f"MC budget={args.budget_ins_del}, p={args.ins_del_p}"
            )

        if daffodil_samples is not None and len(daffodil_samples) > 0:
            ins_del_tasks.append(
                InsDelTask(
                    samples=daffodil_samples,
                    class_names=DAFFODIL_LABELS,
                    prompt_template=lambda c: f"This image shows a case of {c}",
                    explain_classes="top1",
                    budget=args.budget_ins_del,
                    p=args.ins_del_p,
                )
            )
            logger.info(
                f"Standalone ins/del (Daffodil/zero-shot) ready — {len(daffodil_samples)} sample(s).  "
                f"MC budget={args.budget_ins_del}, p={args.ins_del_p}"
            )

        if caption_samples is not None and len(caption_samples) > 0:
            ins_del_tasks.append(
                InsDelTask(
                    samples=caption_samples,
                    budget=args.budget_ins_del,
                    p=args.ins_del_p,
                )
            )
            logger.info(
                f"Standalone ins/del (Derm1M/caption) ready — "
                f"{len(caption_samples)} sample(s).  "
                f"MC budget={args.budget_ins_del}, p={args.ins_del_p}"
            )

        if not ins_del_tasks:
            logger.error(
                "Standalone ins_del requested but both HAM_IMAGES and "
                "DERM1M_ENTRIES are empty. Nothing to do."
            )
            sys.exit(1)

    # --- Run models ---
    all_results: dict = {}

    for cfg in MODELS:
        model_name      = cfg["name"]
        budget_caption  = args.budget_caption  or cfg.get("budget_caption",  2 ** 10)
        budget_zeroshot = args.budget_zeroshot or cfg.get("budget_zeroshot", 2 ** 8)
        loader_kwargs   = {k: v for k, v in cfg.items()
                           if k not in ("budget_caption", "budget_zeroshot")}

        logger.info(f"\n{'─' * 60}")
        logger.info(f"Loading model: {model_name}")
        log_gpu_memory(logger, "before load")

        model    = load_model(**loader_kwargs)
        out_dirs = make_output_dirs(args.output_dir, model_name, tasks)
        all_results[model_name] = {}

        # ins/del output dir (used when combined with caption/zeroshot)
        ins_del_dir = out_dirs.get("ins_del") if do_ins_del else None

        # ------------------------------------------------------------------
        # Caption task  (+ optional ins/del curves from the same IV)
        # ------------------------------------------------------------------
        if caption_task is not None:
            logger.info(
                f"  Running caption task (budget={budget_caption}"
                + (f", ins_del -> {ins_del_dir}" if ins_del_dir else "")
                + ")"
            )
            caption_results = explain(
                model,
                caption_task,
                budget=budget_caption,
                output_dir=out_dirs["caption"],
                ins_del_output_dir=ins_del_dir,
                ins_del_batch_size=args.ins_del_batch_size,
            )
            release_results(caption_results)
            all_results[model_name]["caption"] = caption_results
            logger.info(f"  Caption results saved → {out_dirs['caption']}")

        # ------------------------------------------------------------------
        # Zero-shot task  (+ optional ins/del curves from the same IV)
        # ------------------------------------------------------------------
        if zeroshot_tasks:
            for task_name, task in zeroshot_tasks:
                logger.info(
                    f"  Running zero-shot task {task_name} (budget={budget_zeroshot}"
                    + (f", ins_del -> {ins_del_dir}" if ins_del_dir else "")
                    + ")"
                )
                zs_results = explain(
                    model,
                    task,
                    budget=budget_zeroshot,
                    output_dir=out_dirs["zeroshot"],
                    ins_del_output_dir=ins_del_dir,
                    ins_del_batch_size=args.ins_del_batch_size,
                )
                release_results(zs_results)
                all_results[model_name].setdefault("zeroshot", []).extend(zs_results)
                logger.info(f"  Zero-shot results saved → {out_dirs['zeroshot']} ({task_name})")

        # ------------------------------------------------------------------
        # Standalone ins/del task(s) — run over HAM_IMAGES and/or DERM1M_ENTRIES
        # ------------------------------------------------------------------
        if ins_del_tasks:
            combined_id_results = []
            for idx, id_task in enumerate(ins_del_tasks):
                logger.info(
                    f"  Running standalone ins/del task [{idx + 1}/{len(ins_del_tasks)}] "
                    f"(budget={args.budget_ins_del}, p={args.ins_del_p}, "
                    f"n_samples={len(id_task.samples)})"
                )
                id_results = id_task.run(
                    model=model,
                    output_dir=out_dirs["ins_del"],
                    batch_size=args.ins_del_batch_size,
                )
                combined_id_results.extend(id_results)
                for r in id_results:
                    logger.info(
                        f"    {r['identifier']}: AID={r['aid']:.4f}"
                        f"  (full={r['v_full']:.4f}, empty={r['v_empty']:.4f})"
                    )
            all_results[model_name]["ins_del"] = combined_id_results
            logger.info(f"  Ins/del results saved → {out_dirs['ins_del']}")

        # --- GPU cleanup ---
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        log_gpu_memory(logger, "after cleanup")

    # --- Summary ---
    logger.info("\nAll models done.")
    for model_name, task_dict in all_results.items():
        for task_name, result in task_dict.items():
            if task_name == "ins_del" and isinstance(result, list):
                for r in result:
                    logger.info(
                        f"  [{model_name}]  ins_del  {r.get('identifier', '?')}: "
                        f"AID={r.get('aid', '?'):.4f}"
                    )
            else:
                logger.info(f"  [{model_name}]  {task_name}: {len(result)} item(s)")

    logger.info("Inference run complete.")


if __name__ == "__main__":
    main()