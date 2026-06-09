"""Task definitions: caption explanation vs zero-shot classification."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from PIL import Image
import torch


@dataclass
class SampleInput:
    """A single sample to explain."""
    image: Image.Image
    text: str
    identifier: str  # e.g. filename or dataset index
    metadata: dict = field(default_factory=dict)


@dataclass
class InferenceResult:
    sample: SampleInput
    target_class: Optional[str] = None
    logit: Optional[float] = None
    probability: Optional[float] = None


class Task(ABC):
    """Yields (sample, prediction_info) tuples for the explanation loop."""

    @abstractmethod
    def iter_samples(self, model) -> List[InferenceResult]:
        ...


# ---------- Caption-based task (e.g. Derm1M) ----------
@dataclass
class CaptionTask(Task):
    """Explain the alignment between image and its provided caption."""
    samples: List[SampleInput]

    def iter_samples(self, model) -> List[InferenceResult]:
        return [InferenceResult(sample=s) for s in self.samples]


# ---------- Zero-shot classification task (e.g. HAM10000) ----------
@dataclass
class ZeroShotTask(Task):
    """Run zero-shot classification, optionally explain all classes or top-k."""
    samples: List[SampleInput]  # text field is ignored / replaced
    class_names: List[str]
    prompt_template: Callable[[str], str] = lambda c: f"This is a photo of {c}"
    explain_classes: str = "all"  # "all" | "top1" | "topk"
    top_k: int = 3

    def iter_samples(self, model) -> List[InferenceResult]:
        device = model.device
        prompts = [self.prompt_template(c) for c in self.class_names]

        # Encode text prompts once
        with torch.no_grad(), torch.autocast(device):
            if model.backend == "huggingface":
                enc = model.tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                text_feats = model.encode_text(enc)
            else:
                enc = model.tokenizer(prompts).to(device)
                text_feats = model.encode_text(enc)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        results = []
        for sample in self.samples:
            img_tensor = model.processor(sample.image).unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast(device):
                img_feat = model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                logits = (100.0 * img_feat @ text_feats.T).squeeze(0)
                probs = logits.softmax(dim=-1)

            # Decide which classes to explain
            if self.explain_classes == "all":
                indices = range(len(self.class_names))
            elif self.explain_classes == "top1":
                indices = [int(torch.argmax(probs).item())]
            else:  # topk
                indices = torch.topk(probs, self.top_k).indices.tolist()

            print(f"\n--- Predictions for {sample.identifier} ---")
            for c_idx, cname in enumerate(self.class_names):
                print(f"  {cname:<25}: logit={logits[c_idx]:.3f} "
                      f"prob={probs[c_idx]:.4f}")

            for c_idx in indices:
                cname = self.class_names[c_idx]
                # Replace text with the class-specific prompt
                new_sample = SampleInput(
                    image=sample.image,
                    text=self.prompt_template(cname),
                    identifier=sample.identifier,
                    metadata=sample.metadata,
                )
                results.append(InferenceResult(
                    sample=new_sample,
                    target_class=cname,
                    logit=float(logits[c_idx].item()),
                    probability=float(probs[c_idx].item()),
                ))
        return results