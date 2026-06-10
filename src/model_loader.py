"""Unified model loading interface for HuggingFace and OpenCLIP backends."""
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
import torch
import open_clip
from transformers import AutoModel, AutoProcessor, AutoTokenizer


# Default CLIP normalization stats
DEFAULT_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
DEFAULT_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class LoadedModel:
    """Container for a loaded vision-language model."""
    name: str
    backend: str  # "huggingface" | "open_clip"
    model: Any
    processor: Any  # image preprocessing
    tokenizer: Any
    image_mean: Tuple[float, float, float]
    image_std: Tuple[float, float, float]
    patch_size: int
    device: str

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.backend == "huggingface":
            return self.model.get_image_features(pixel_values)
        return self.model.encode_image(pixel_values)

    def encode_text(self, tokens) -> torch.Tensor:
        if self.backend == "huggingface":
            return self.model.get_text_features(**tokens)
        return self.model.encode_text(tokens)


class _CustomHFTokenizer:
    """Wraps an HF tokenizer to act like an open_clip tokenizer."""
    def __init__(self, hf_tokenizer, max_length: int = 256):
        self.tk = hf_tokenizer
        self.max_length = max_length

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.tk(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        ).input_ids

    def decode(self, token_ids):
        return self.tk.decode(token_ids)


def _infer_patch_size(model) -> int:
    if hasattr(model, "visual") and hasattr(model.visual, "patch_size"):
        ps = model.visual.patch_size
        return ps[0] if isinstance(ps, tuple) else ps
    return 16


def load_model(
    name: str,
    backend: str,
    pretrained: Optional[str] = None,
    device: str = "cuda",
    hf_tokenizer_name: Optional[str] = None,
) -> LoadedModel:
    """Load a CLIP-like model from either HuggingFace or OpenCLIP."""
    if backend == "huggingface":
        model = AutoModel.from_pretrained(name).to(device).eval()
        try:
            processor = AutoProcessor.from_pretrained(name, use_fast=False)
        except TypeError:
            processor = AutoProcessor.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        return LoadedModel(
            name=name,
            backend=backend,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image_mean=tuple(processor.image_processor.image_mean),
            image_std=tuple(processor.image_processor.image_std),
            patch_size=_infer_patch_size(model),
            device=device,
        )

    if backend == "open_clip":
        if pretrained:
            model, _, processor = open_clip.create_model_and_transforms(
                name, pretrained=pretrained
            )
        else:
            model, _, processor = open_clip.create_model_and_transforms(name)
        model = model.to(device).eval()

        if hf_tokenizer_name:
            tokenizer = _CustomHFTokenizer(
                AutoTokenizer.from_pretrained(hf_tokenizer_name)
            )
        else:
            tokenizer = open_clip.get_tokenizer(name)

        return LoadedModel(
            name=name,
            backend=backend,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image_mean=DEFAULT_IMAGE_MEAN,
            image_std=DEFAULT_IMAGE_STD,
            patch_size=_infer_patch_size(model),
            device=device,
        )

    raise ValueError(f"Unknown backend: {backend}")