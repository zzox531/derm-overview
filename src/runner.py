"""Runs FIxLIP explanations across (model, task) combinations.

Changes from original
---------------------
* After ``approximator.approximate_crossmodal()`` the interaction values are
  optionally forwarded to ``src.ins_del.run_and_save()`` when the caller
  passes ``ins_del_output_dir``.  This is the only structural change; all
  existing behaviour is preserved when the argument is omitted.
"""
import gc
import torch
import matplotlib.pyplot as plt
import src.utils
import src.plot
import src.fixlip
import src.game_huggingface
import src.game_openclip
from src.model_loader import _CustomHFTokenizer
from src.model_loader import LoadedModel
from src.tasks import Task, InferenceResult
from pathlib import Path
from typing import Optional


def _build_game(model: LoadedModel, sample, batch_size: int = 32):
    if model.backend == "huggingface":
        return src.game_huggingface.VisionLanguageGame(
            model=model.model,
            processor=model.processor,
            input_image=sample.image,
            input_text=sample.text,
            batch_size=batch_size,
        )
    return src.game_openclip.OpenCLIPGame(
        model=model.model,
        image_processor=model.processor,
        text_tokenizer=model.tokenizer,
        input_image=sample.image,
        input_text=sample.text,
        patch_size=model.patch_size,
        batch_size=batch_size,
    )


def _extract_text_tokens(model: LoadedModel, game, input_text: str):
    if model.backend == "huggingface":
        ids = game.inputs["input_ids"][0]
        if hasattr(model.tokenizer, "convert_ids_to_tokens"):
            tokens = model.tokenizer.convert_ids_to_tokens(ids)
        else:
            tokens = [model.tokenizer.decode([t.item()]) for t in ids]
    else:
        token_ids = model.tokenizer(input_text)[0]
                    

        if isinstance(model.tokenizer, _CustomHFTokenizer):
            # Returns list[str] directly — one token per id
            tokens = model.tokenizer.decode(token_ids.tolist())
            token_ids = model.tokenizer(input_text)[0]
            raw = model.tokenizer.tk.convert_ids_to_tokens(
                [int(i) for i in token_ids], skip_special_tokens=False
            )
            print("RAW token ids:", token_ids[:20].tolist())
            print("RAW tokens:   ", raw[:20])
            tokens = model.tokenizer.decode(token_ids.tolist())
            print("After decode: ", tokens)
            return [
                t.replace("</w>", "")
                 .replace("▁", " ")
                 .replace("<|startoftext|>", "")
                 .replace("<|endoftext|>", "")
                 .strip()
                for t in tokens
                if isinstance(t, str) and t not in ("<pad>", "</s>", "<s>")
            ]
        elif hasattr(model.tokenizer, "decode"):
            tokens = [model.tokenizer.decode([t.item()]) for t in token_ids]
        else:
            tokens = [str(t) for t in token_ids.numpy().tolist()]

    if len(tokens) > game.n_players_text:
        tokens = tokens[1: game.n_players_text + 1]

    return [
        t.replace("</w>", "")
         .replace("▁", " ")
         .replace("<|startoftext|>", "")
         .replace("<|endoftext|>", "")
         .strip()
        for t in tokens
        if isinstance(t, str) and t not in ("<pad>", "</s>", "<s>")
    ]


def explain(
    model: LoadedModel,
    task: Task,
    *,
    budget: int = 2 ** 10,
    max_order: int = 2,
    p: float = 0.5,
    max_text_players: int = 30,
    random_state: int = 42,
    top_k_plot: int = 13,
    plot: bool = True,
    output_dir=None,
    # --- new: ins/del ---
    ins_del_output_dir: Optional[Path] = None,
    ins_del_batch_size: int = 64,
):
    """Run FIxLIP explanations for every (sample, target) yielded by the task.

    Parameters (additions only — all originals unchanged)
    -------------------------------------------------------
    ins_del_output_dir : Path or None
        When provided, insertion/deletion curves are computed from the FIxLIP
        interaction values and saved to this directory after each sample.
        Pass ``None`` (default) to skip the experiment entirely.
    ins_del_batch_size : int
        Batch size for the masked-image scoring inside the ins/del sweep.
    """
    # Lazy import so that the module is only required when ins/del is active.
    if ins_del_output_dir is not None:
        import src.ins_del as _ins_del
        ins_del_output_dir = Path(ins_del_output_dir)
        ins_del_output_dir.mkdir(parents=True, exist_ok=True)

    out = []
    for item in task.iter_samples(model):
        sample = item.sample
        title_suffix = (
            f" | class='{item.target_class}'"
            f" prob={item.probability:.3f}"
            if item.target_class else ""
        )
        print(f"\n--> {sample.identifier}{title_suffix}")

        game = _build_game(model, sample)
        if game.n_players_text > max_text_players:
            print(f"   [skip] too many text players: {game.n_players_text}")
            continue

        approximator = src.fixlip.FIxLIP(
            n_players_text=game.n_players_text,
            n_players_image=game.n_players_image,
            max_order=max_order,
            p=p,
            random_state=random_state,
        )
        iv = approximator.approximate_crossmodal(game=game, budget=budget)
        item_result = {
            "identifier": sample.identifier,
            "target_class": item.target_class,
            "logit": item.logit,
            "probability": item.probability,
            "interaction_values": iv,
        }
        out.append(item_result)

        # ------------------------------------------------------------------ #
        # Insertion / deletion experiment (new — only when dir is given)      #
        # ------------------------------------------------------------------ #
        if ins_del_output_dir is not None:
            _ins_del.run_and_save(
                item=item,
                model=model,
                iv=iv,
                output_dir=ins_del_output_dir,
                batch_size=ins_del_batch_size,
                # budget / p / random_state only used if iv is None,
                # but passed for completeness
                budget=budget,
                p=p,
                random_state=random_state,
            )

        # ------------------------------------------------------------------ #
        # Original saliency plotting (unchanged)                              #
        # ------------------------------------------------------------------ #
        if plot:
            tokens = _extract_text_tokens(model, game, sample.text)
            if model.backend == "huggingface":
                img_t = game.inputs["pixel_values"].squeeze(0)
            else:
                img_t = game.inputs[0].squeeze(0)
            img_np = src.utils.denormalize(
                img_t, model.image_mean, model.image_std
            ).permute(1, 2, 0).numpy()

            print("n_players_image:", game.n_players_image)
            print("n_players_text:", game.n_players_text)
            print("iv.n_players:", iv.n_players)
            print("iv index:", iv.index)
            # Check the actual interaction indices stored
            all_players = set(p for interaction in iv.interaction_lookup for p in interaction)
            print("player index range in iv:", min(all_players), "to", max(all_players))
            
            src.plot.plot_image_and_text_together(
                img=img_np,
                text=tokens,
                image_players=list(range(game.n_players_image)),
                iv=iv,
                plot_interactions=True,
                top_k=top_k_plot,
                normalize_jointly=True,
                figsize=(8, 8),
                fontsize=16,
                margin=0.3,
                color_text=True,
                plot_heatmap=True,
                show=True,
                max_value=float(iv.values.max() * 4),
            )
            if item.target_class:
                plt.title(f"'{item.target_class}'", pad=20)
            plt.tight_layout(pad=0.15)

            if output_dir is not None:
                out_path = Path(output_dir) / f"{sample.identifier.replace('/', '_')}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_path, bbox_inches="tight", dpi=150)
            else:
                plt.show()
            plt.close("all")

            plt.close("all")

        del game, approximator, iv
        gc.collect()
        torch.cuda.empty_cache()
    return out