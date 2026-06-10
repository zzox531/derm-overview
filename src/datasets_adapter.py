"""Adapters that convert various datasets into a list of SampleInput."""
from pathlib import Path
from typing import List, Optional
from PIL import Image
from src.tasks import SampleInput


def from_derm1m(ds, entries: List[dict], image_root: str) -> List[SampleInput]:
    """entries: list of {"filename": str, "index": int}."""
    samples = []
    for e in entries:
        path = Path(image_root) / e["filename"]
        if not path.exists():
            print(f"[!] Missing: {path}")
            continue
        sample = ds["train"][e["index"]]
        samples.append(SampleInput(
            image=Image.open(path).convert("RGB"),
            text=sample["caption"],
            identifier=str(path),
            metadata={"disease_label": sample.get("disease_label")},
        ))
    return samples


def from_image_folder(
    files: List[str],
    default_text: str = "",
) -> List[SampleInput]:
    """Generic folder loader for zero-shot use cases (e.g. HAM10000)."""
    samples = []
    for f in files:
        path = Path(f)
        if not path.exists():
            print(f"[!] Missing: {path}")
            continue
        samples.append(SampleInput(
            image=Image.open(path).convert("RGB"),
            text=default_text,
            identifier=str(path),
        ))
    return samples