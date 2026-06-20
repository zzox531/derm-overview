"""
src/ins_del.py — Insertion / Deletion experiment for vision-language models.
-------
AID  (Insertion vs Deletion):
    mean(insertion_MIF_norm - deletion_MIF_norm)
    Higher is better.
"""

from __future__ import annotations

import csv
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from src.tasks import InferenceResult, SampleInput, Task

if TYPE_CHECKING:
    from src.model_loader import LoadedModel

class ImagePatcher:
    def __init__(
        self,
        image: Image.Image,
        patch_size: int = 16,
        model_size: int = 224,
        fill_color: tuple = (128, 128, 128),
    ) -> None:
        self.patch_size = patch_size
        self.model_size = model_size
        self.grey = np.array(fill_color, dtype=np.uint8)

        resized = image.convert("RGB").resize((model_size, model_size), Image.BICUBIC)
        self.img_array = np.array(resized, dtype=np.uint8)

        self.n_per_side = model_size // patch_size
        self.n_patches  = self.n_per_side ** 2

    def _coords(self, idx: int) -> tuple[int, int, int, int]:
        r, c = divmod(idx, self.n_per_side)
        return (
            r * self.patch_size, (r + 1) * self.patch_size,
            c * self.patch_size, (c + 1) * self.patch_size,
        )

    def mask_image(self, active: np.ndarray) -> Image.Image:
        out = self.img_array.copy()
        for i, on in enumerate(active):
            if not on:
                r0, r1, c0, c1 = self._coords(i)
                out[r0:r1, c0:c1] = self.grey
        return Image.fromarray(out)

    def build_images(self, coalition_matrix: np.ndarray) -> list[Image.Image]:
        return [self.mask_image(row) for row in coalition_matrix]

    @property
    def empty_coalition(self) -> np.ndarray:
        return np.zeros(self.n_patches, dtype=bool)

@torch.no_grad()
def score_coalitions(
    images: list[Image.Image],
    text: str,
    model: "LoadedModel",
    batch_size: int = 64,
) -> np.ndarray:
    device = model.device
    all_scores: list[np.ndarray] = []

    for start in range(0, len(images), batch_size):
        batch = images[start: start + batch_size]

        if model.backend == "huggingface":
            inputs = model.processor(
                text=[text],
                images=batch,
                return_tensors="pt",
                padding=True,
            ).to(device)
            out = model.model(**inputs)
            img_emb = out.image_embeds
            txt_emb = out.text_embeds

        else:
            import open_clip
            img_tensor = torch.stack(
                [model.processor(img) for img in batch]
            ).to(device)
            tok = model.tokenizer([text]).to(device)
            img_emb = model.model.encode_image(img_tensor)
            txt_emb = model.model.encode_text(tok)

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ txt_emb.T).squeeze(-1).float().cpu().numpy()
        all_scores.append(sims)

    return np.concatenate(all_scores)

def _build_mif_del_from_attributions(
    attribution_values: np.ndarray,
    empty_coalition: np.ndarray,
) -> np.ndarray:
    """Build MIF deletion coalition matrix from 1-D attribution values."""
    attribution_values = np.asarray(attribution_values).reshape(-1)

    if len(empty_coalition) != len(attribution_values):
        raise ValueError(
            f"Shape mismatch: empty_coalition={len(empty_coalition)} "
            f"vs attribution_values={len(attribution_values)}"
        )

    n = len(attribution_values)
    order = np.argsort(attribution_values)[::-1]

    rows = [np.ones(n, dtype=bool)]
    active = np.ones(n, dtype=bool)
    for idx in order:
        active = active.copy()
        active[idx] = False
        rows.append(active.copy())

    rows[-1] = np.asarray(empty_coalition, dtype=bool).reshape(-1)

    return np.stack(rows, axis=0)

def _build_mif_ins_from_attributions(
    attribution_values: np.ndarray,
) -> np.ndarray:
    """Build MIF insertion coalition matrix from 1-D attribution values.

    Insertion MIF: starts with the empty coalition, then adds highest-attribution
    patches first. Row 0 is the empty coalition; row i+1 has the top-i patches
    active. Final row is the full coalition (all patches active).
    """
    attribution_values = np.asarray(attribution_values).reshape(-1)
    n = len(attribution_values)

    order = np.argsort(attribution_values)[::-1]

    rows = [np.zeros(n, dtype=bool)]
    active = np.zeros(n, dtype=bool)
    for idx in order:
        active = active.copy()
        active[idx] = True
        rows.append(active.copy())

    rows[-1] = np.ones(n, dtype=bool)

    return np.stack(rows, axis=0)

def compute_mc_attributions(
    patcher: ImagePatcher,
    text: str,
    model: "LoadedModel",
    budget: int = 1024,
    p: float = 0.5,
    batch_size: int = 64,
    random_state: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = patcher.n_patches
    n_rounds = max(1, budget // (2 * n))

    print(
        f"  [ins_del] MC attribution: {n_rounds} rounds × {n} players × 2 "
        f"= {2 * n * n_rounds} evals",
        flush=True,
    )

    marginals = np.zeros(n, dtype=np.float64)

    for _ in range(n_rounds):
        base = rng.random(n) < p

        imgs_on, imgs_off = [], []
        for i in range(n):
            c_on  = base.copy(); c_on[i]  = True
            c_off = base.copy(); c_off[i] = False
            imgs_on.append(patcher.mask_image(c_on))
            imgs_off.append(patcher.mask_image(c_off))

        s_on  = score_coalitions(imgs_on,  text, model, batch_size)
        s_off = score_coalitions(imgs_off, text, model, batch_size)
        marginals += s_on - s_off

    return marginals / n_rounds

def run_insertion_deletion(
    patcher: ImagePatcher,
    text: str,
    model: "LoadedModel",
    attribution_values: np.ndarray,
    batch_size: int = 64,
) -> dict:
    n = patcher.n_patches
    empty_coalition = patcher.empty_coalition

    coalition_matrix_del = _build_mif_del_from_attributions(
        attribution_values,
        empty_coalition,
    )

    coalition_matrix_ins = _build_mif_ins_from_attributions(attribution_values)

    assert coalition_matrix_del[-1].sum() == 0, \
        "Deletion must end at empty coalition"
    assert coalition_matrix_ins[-1].sum() == n, \
        "Insertion must end at full coalition"

    def _score(coalition_matrix: np.ndarray) -> np.ndarray:
        images = patcher.build_images(coalition_matrix)
        return score_coalitions(images, text, model, batch_size)

    print("  [ins_del] Scoring deletion-MIF...", flush=True)
    predictions_del = _score(coalition_matrix_del)

    print("  [ins_del] Scoring insertion-MIF...", flush=True)
    predictions_ins = _score(coalition_matrix_ins)

    v_full  = float(predictions_del[0])
    v_empty = float(predictions_del[-1])
    denom = v_full - v_empty if abs(v_full - v_empty) > 1e-8 else 1.0

    def _norm(arr: np.ndarray) -> np.ndarray:
        return (arr - v_empty) / denom

    predictions_del_norm = _norm(predictions_del)
    predictions_ins_norm = _norm(predictions_ins)

    fractions_del = coalition_matrix_del.sum(axis=1) / n
    fractions_ins = coalition_matrix_ins.sum(axis=1) / n

    interp_ins = np.interp(fractions_del, fractions_ins, predictions_ins_norm)
    aid = float(np.mean(interp_ins - predictions_del_norm))

    return {
        "order": 1,
        "fractions_del_mif": fractions_del,
        "fractions_ins_mif": fractions_ins,
        "deletion_mif": predictions_del_norm,
        "insertion_mif": predictions_ins_norm,
        "aid": aid,
        "v_full": v_full,
        "v_empty": v_empty,
    }

def run_insertion_deletion_order2(
    patcher: ImagePatcher,
    text: str,
    model: "LoadedModel",
    iv,
    p_sampler: float = 0.5,
    batch_size: int = 64,
) -> dict:
    import src.utils

    n = patcher.n_patches
    empty_coalition = patcher.empty_coalition

    iv_first_order = src.utils.convert_iv_to_first_order(iv, p_sampler=p_sampler)
    baseline_attribution = _extract_image_attributions(iv_first_order, n_image_patches=n)

    coalition_matrix_del = _build_mif_del_from_attributions(baseline_attribution, empty_coalition)

    coalition_matrix_ins = _build_mif_ins_from_attributions(baseline_attribution)

    def _score(coalition_matrix: np.ndarray) -> np.ndarray:
        images = patcher.build_images(coalition_matrix)
        return score_coalitions(images, text, model, batch_size)

    print("  [ins_del] Scoring deletion-MIF (order-2)...", flush=True)
    predictions_del = _score(coalition_matrix_del)

    print("  [ins_del] Scoring insertion-MIF (order-2)...", flush=True)
    predictions_ins = _score(coalition_matrix_ins)

    v_full  = float(predictions_del[0])
    v_empty = float(predictions_del[-1])
    denom = v_full - v_empty if abs(v_full - v_empty) > 1e-8 else 1.0

    def _norm(arr: np.ndarray) -> np.ndarray:
        return (arr - v_empty) / denom

    predictions_del_norm = _norm(predictions_del)
    predictions_ins_norm = _norm(predictions_ins)

    fractions_del = coalition_matrix_del.sum(axis=1) / n
    fractions_ins = coalition_matrix_ins.sum(axis=1) / n

    interp_ins = np.interp(fractions_del, fractions_ins, predictions_ins_norm)
    aid = float(np.mean(interp_ins - predictions_del_norm))

    return {
        "order": 2,
        "fractions_del_mif": fractions_del,
        "fractions_ins_mif": fractions_ins,
        "deletion_mif": predictions_del_norm,
        "insertion_mif": predictions_ins_norm,
        "aid": aid,
        "v_full": v_full,
        "v_empty": v_empty,
    }

def plot_curves(
    results: dict,
    title: str = "",
    output_path: Optional[str] = None,
    show: bool = False,
    smooth_sigma: float = 1.0,
) -> None:
    """Plot insertion and deletion curves.

    X-axis is "percentage of patches retained", reversed so that:
      - Deletion runs left→right (100% → 0%): removes most important first
      - Insertion runs right→left (0% → 100%): adds most important first
    """
    order = results.get("order", 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    def _smooth_and_pin(arr: np.ndarray, start_val: float, end_val: float) -> np.ndarray:
        if smooth_sigma > 0:
            arr = gaussian_filter1d(arr.astype(float), sigma=smooth_sigma)
            arr[0]  = start_val
            arr[-1] = end_val
        return arr

    frac_del = results["fractions_del_mif"] * 100   # 100% → 0%
    frac_ins = results["fractions_ins_mif"] * 100   # 0%   → 100%

    del_arr = results["deletion_mif"].astype(float)
    ins_arr = results["insertion_mif"].astype(float)

    del_curve = _smooth_and_pin(del_arr, del_arr[0], del_arr[-1])
    ins_curve = _smooth_and_pin(ins_arr, ins_arr[0], ins_arr[-1])

    label_suffix = " (2nd order)" if order == 2 else ""

    ax.plot(
        frac_del,
        del_curve,
        color="#e05c2a",
        lw=2,
        ls="--",
        label=(
            f"Deletion MIF{label_suffix} "
            f"(lower=better)  "
            f"AID={results['aid']:.3f}"
        ),
    )

    ax.plot(
        frac_ins,
        ins_curve,
        color="#2a7ae0",
        lw=2,
        ls="-",
        label=f"Insertion MIF{label_suffix} (higher=better)",
    )

    ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
    ax.axhline(1, color="black", lw=0.5, ls="--", alpha=0.4)

    ax.set_xlim(100, 0)

    ax.set_xlabel("Percentage of patches retained (%)", fontsize=12)
    ax.set_ylabel("Prediction change (normalised)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close("all")

def save_results_csv(results: dict, output_path: str) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fraction_del_mif",
            "deletion_mif",
            "fraction_ins_mif",
            "insertion_mif",
        ])
        writer.writerows(zip(
            results["fractions_del_mif"],
            results["deletion_mif"],
            results["fractions_ins_mif"],
            results["insertion_mif"],
        ))

def _extract_image_attributions(iv, n_image_patches: int) -> np.ndarray:
    if hasattr(iv, "get_n_order"):
        # Index by player key explicitly — avoids assuming a particular ordering of .values.
        try:
            return np.array([iv[(i,)] for i in range(n_image_patches)])
        except KeyError as exc:
            raise ValueError(
                f"Image patch player {exc} not found in interaction values."
            ) from exc

    arr = np.asarray(iv)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D attribution array, got shape {arr.shape}.")
    if len(arr) < n_image_patches:
        raise ValueError(
            f"Attribution array has {len(arr)} entries "
            f"but image has {n_image_patches} patches."
        )
    return arr[:n_image_patches].copy()

def _resolve_model_size(model: "LoadedModel") -> int:
    try:
        if model.backend == "huggingface":
            size = model.processor.image_processor.size
            for key in ("shortest_edge", "height", "size"):
                if key in size:
                    val = size[key]
                    return int(val[0]) if isinstance(val, (list, tuple)) else int(val)
            return int(min(v if isinstance(v, int) else v[0] for v in size.values()))
        else:
            val = model.model.visual.image_size
            return int(val[0]) if isinstance(val, (list, tuple)) else int(val)
    except Exception:
        return 224

def _get_iv_order(iv) -> int:
    if hasattr(iv, "max_order"):
        return int(iv.max_order)
    if hasattr(iv, "get_n_order"):
        try:
            if len(iv.get_n_order(2).values) > 0:
                return 2
        except Exception:
            pass
    return 1

def _get_p_sampler(iv) -> float:
    if hasattr(iv, "sampling_weights"):
        w = np.asarray(iv.sampling_weights)
        if len(w) > 0:
            return float(w[0])
    return 0.5

def run_and_save(
    *,
    item: InferenceResult,
    model: "LoadedModel",
    iv,
    output_dir: Path,
    batch_size: int = 64,
    budget: int = 1024,
    p: float = 0.5,
    random_state: int = 42,
) -> dict:
    sample = item.sample

    patch_size = getattr(model, "patch_size", 16)
    model_size = _resolve_model_size(model)

    fill_color = tuple(int(m * 255) for m in model.image_mean)
    patcher = ImagePatcher(sample.image, patch_size=patch_size, model_size=model_size,
                           fill_color=fill_color)
    n = patcher.n_patches

    iv_order = _get_iv_order(iv) if iv is not None else 1

    if iv is None:
        attribution_values = compute_mc_attributions(
            patcher=patcher,
            text=sample.text,
            model=model,
            budget=budget,
            p=p,
            batch_size=batch_size,
            random_state=random_state,
        )
        attr_path = output_dir / f"{Path(sample.identifier).with_suffix('').as_posix().replace('/', '_')}_attr.npy"
        np.save(attr_path, attribution_values)
        results = run_insertion_deletion(
            patcher=patcher,
            text=sample.text,
            model=model,
            attribution_values=attribution_values,
            batch_size=batch_size,
        )

    elif iv_order == 1:
        attribution_values = _extract_image_attributions(iv, n_image_patches=n)
        results = run_insertion_deletion(
            patcher=patcher,
            text=sample.text,
            model=model,
            attribution_values=attribution_values,
            batch_size=batch_size,
        )

    else:
        p_sampler = _get_p_sampler(iv)
        results = run_insertion_deletion_order2(
            patcher=patcher,
            text=sample.text,
            model=model,
            iv=iv,
            p_sampler=p_sampler,
            batch_size=batch_size,
        )

    order = results["order"]

    pred_class = (item.target_class or "").replace(" ", "_")
    img_stem = Path(sample.identifier).stem
    stem = f"{img_stem}_pred_{pred_class}"

    csv_path  = output_dir / f"{stem}_curves.csv"
    plot_path = output_dir / f"{stem}_curves.png"

    save_results_csv(results, str(csv_path))

    title_lines = [f"Insertion / Deletion — {sample.identifier}"]
    if item.target_class:
        title_lines.append(
            f"class='{item.target_class}'  prob={item.probability:.3f}"
        )
    title_lines.append(f"'{sample.text[:70]}'")

    plot_curves(
        results,
        title="\n".join(title_lines),
        output_path=str(plot_path),
    )

    aid_str = f"  AID={results['aid']:.4f}" if "aid" in results else ""
    print(
        f"  [ins_del] order={order}{aid_str}"
        f"  (full={results['v_full']:.4f}, empty={results['v_empty']:.4f})"
        f"  → {plot_path.name}",
        flush=True,
    )

    return results

@dataclass
class InsDelTask(Task):
    samples: List[SampleInput]
    class_names: Optional[List[str]] = None
    prompt_template: Callable[[str], str] = field(
        default=lambda c: f"This is a photo of {c}"
    )
    explain_classes: str = "top1"
    top_k: int = 3
    budget: int = 1024
    p: float = 0.5
    random_state: int = 42

    def iter_samples(self, model: "LoadedModel") -> List[InferenceResult]:
        if not self.class_names:
            return [InferenceResult(sample=s) for s in self.samples]

        device = model.device
        prompts = [self.prompt_template(c) for c in self.class_names]

        with torch.no_grad():
            if model.backend == "huggingface":
                enc = model.tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                text_feats = model.model.get_text_features(**enc)
            else:
                enc = model.tokenizer(prompts).to(device)
                text_feats = model.model.encode_text(enc)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        results: List[InferenceResult] = []
        for sample in self.samples:
            if model.backend == "huggingface":
                img_inputs = model.processor(
                    images=sample.image, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    img_feat = model.model.get_image_features(**img_inputs)
            else:
                img_tensor = model.processor(sample.image).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_feat = model.model.encode_image(img_tensor)

            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = (model.get_logit_scale() * img_feat @ text_feats.T).squeeze(0)
            probs  = logits.softmax(dim=-1)

            if self.explain_classes == "all":
                indices = list(range(len(self.class_names)))
            elif self.explain_classes == "top1":
                indices = [int(torch.argmax(probs).item())]
            else:
                indices = torch.topk(probs, self.top_k).indices.tolist()

            for c_idx in indices:
                cname = self.class_names[c_idx]
                results.append(InferenceResult(
                    sample=SampleInput(
                        image=sample.image,
                        text=self.prompt_template(cname),
                        identifier=sample.identifier,
                        metadata=sample.metadata,
                    ),
                    target_class=cname,
                    logit=float(logits[c_idx].item()),
                    probability=float(probs[c_idx].item()),
                ))

        return results

    def run(
        self,
        model: "LoadedModel",
        output_dir: Path,
        batch_size: int = 64,
    ) -> List[dict]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        all_results = []
        for item in self.iter_samples(model):
            result = run_and_save(
                item=item,
                model=model,
                iv=item.sample.metadata.get("iv"),
                output_dir=output_dir,
                batch_size=batch_size,
                budget=self.budget,
                p=self.p,
                random_state=self.random_state,
            )
            all_results.append({"identifier": item.sample.identifier, **result})
        return all_results
