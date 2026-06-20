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
            return _unwrap_hf_output(self.model.get_image_features(pixel_values))
        return self.model.encode_image(pixel_values)

    def encode_text(self, tokens) -> torch.Tensor:
        if self.backend == "huggingface":
            return _unwrap_hf_output(self.model.get_text_features(**tokens))
        return self.model.encode_text(tokens)

    def get_logit_scale(self) -> float:
        try:
            return float(self.model.logit_scale.exp().item())
        except AttributeError:
            return 100.0
    
    def decode_tokens(self, token_ids) -> list[str]:
        """Decode a sequence of token IDs to per-token strings."""
        if isinstance(self.tokenizer, _CustomHFTokenizer):
            return [self.tokenizer.tk.convert_ids_to_tokens(int(i)) for i in token_ids]

        # Bare HF tokenizer (CLIPTokenizer / CLIPTokenizerFast for monet etc.)
        # open_clip tokenizers don't have convert_ids_to_tokens, so this is safe.
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            return list(self.tokenizer.convert_ids_to_tokens([int(i) for i in token_ids]))

        # open_clip tokenizer — check if it's SentencePiece or BPE
        inner = getattr(self.tokenizer, "tokenizer", None)

        if inner is not None and hasattr(inner, "decoder"):
            # SentencePiece (SigLIP, etc.)
            return [inner.decoder.sp_model.id_to_piece(int(i)) for i in token_ids]

        # Standard BPE (original ViT-B-16, CoCa, etc.)
        return [
            self.tokenizer.decoder.get(int(i), str(int(i)))
            for i in token_ids
        ]


def _unwrap_hf_output(result) -> torch.Tensor:
    """Extract a plain tensor from HF model outputs that may return dataclasses."""
    if isinstance(result, torch.Tensor):
        return result
    if hasattr(result, "pooler_output") and result.pooler_output is not None:
        return result.pooler_output
    if hasattr(result, "last_hidden_state"):
        return result.last_hidden_state[:, 0]
    raise TypeError(f"Cannot extract tensor from {type(result)}")


class _CustomHFTokenizer:
    """Wraps an HF tokenizer to act like an open_clip tokenizer."""
    def __init__(self, hf_tokenizer, max_length: int = 256):
        self.tk = hf_tokenizer
        self.max_length = max_length
        # Detect subword scheme from vocabulary samples
        sample_vocab = list(getattr(self.tk, "vocab", {}).keys())[:500]
        self._is_wordpiece = any(t.startswith("##") for t in sample_vocab)
        self._is_sentencepiece = any(t.startswith("▁") for t in sample_vocab)

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

    def decode_words(self, token_ids) -> list[str]:
        """Decode token IDs, merging continuation subwords into whole words."""
        subwords = self.decode(token_ids)
        words = []
        current = ""
        for t in subwords:
            if t.startswith("·"):
                current += t[1:]
            else:
                if current:
                    words.append(current)
                current = t
        if current:
            words.append(current)
        return words

    def decode(self, token_ids) -> list[str]:
        raw_tokens = self.tk.convert_ids_to_tokens(
            [int(i) for i in token_ids],
            skip_special_tokens=True,
        )
        _SKIP = {"<pad>", "</s>", "<s>", "[CLS]", "[SEP]", "▁",
                 "<|startoftext|>", "<|endoftext|>"}
        result = []
        # Track standalone ▁ space separators (T5/SigLIP tokenizer style).
        # When a content token immediately follows a bare ▁, it starts a new
        # word even though it has no ▁ prefix, so it must NOT get the "·" mark.
        after_space = False
        for t in raw_tokens:
            if not isinstance(t, str) or t in _SKIP or not t.strip():
                after_space = (t == "▁")
                continue
            if t.startswith("##"):
                # WordPiece continuation token — strip ## and mark with ·
                text = t[2:].strip()
                if text:
                    result.append(f"·{text}")
            elif t.startswith("▁"):
                # SentencePiece word-initial token — strip leading ▁
                text = t[1:].replace("</w>", "").strip()
                if text:
                    result.append(text)
            elif self._is_sentencepiece and not after_space:
                # SentencePiece word-continuation (no ▁ prefix, not after bare ▁)
                text = t.replace("</w>", "").strip()
                if text:
                    result.append(f"·{text}")
            else:
                # Word-initial after a standalone ▁ separator, or plain BPE token
                text = t.replace("</w>", "").strip()
                if text:
                    result.append(text)
            after_space = False
        return result


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
    hf_tokenizer_max_length: int = 256,
) -> LoadedModel:
    """Load a CLIP-like model from either HuggingFace or OpenCLIP."""
    if backend == "huggingface":
        model = AutoModel.from_pretrained(name).to(device).eval()
        processor = AutoProcessor.from_pretrained(name)
        text_tokenizer = getattr(processor, "tokenizer", processor)
        return LoadedModel(
            name=name,
            backend=backend,
            model=model,
            processor=processor,
            tokenizer=text_tokenizer,
            image_mean=tuple(processor.image_processor.image_mean),
            image_std=tuple(processor.image_processor.image_std),
            patch_size=_infer_patch_size(model),
            device=device,
        )

    if backend == "open_clip":
        if name.startswith("hf-hub:"):
            model, _, processor = open_clip.create_model_and_transforms(name)
            tokenizer = open_clip.get_tokenizer(name)
        elif pretrained:
            # openai pretrained weights use QuickGELU; force it to avoid mismatch warning
            force_quick_gelu = pretrained.lower() == "openai"
            model, _, processor = open_clip.create_model_and_transforms(
                name, pretrained=pretrained, force_quick_gelu=force_quick_gelu
            )
        else:
            model, _, processor = open_clip.create_model_and_transforms(name)
        model = model.to(device).eval()

        if hf_tokenizer_name:
            tokenizer = _CustomHFTokenizer(
                AutoTokenizer.from_pretrained(hf_tokenizer_name),
                max_length=hf_tokenizer_max_length,
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