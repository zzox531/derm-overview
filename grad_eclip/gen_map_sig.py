import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from open_clip import create_model_from_pretrained, get_tokenizer
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Architecture note
# ---------------------------------------------------------------------------
# ViT-B-16-SigLIP-256 is structurally different from OpenAI's CLIP ViT-B/16
# in one critical way: it has NO CLS token. Its vision backbone (a plain
# timm VisionTransformer) outputs only patch tokens, and the single "global
# image embedding" is produced by a learned-latent cross-attention pooling
# head ("MAP head" / `attn_pool`): one learnable query vector attends over
# all patch tokens, exactly once. There is no stack of CLS self-attention
# layers to pick "the last n" from, so this module plays the same role that
# the CLS token's final self-attention layer(s) played in the original
# CLIP-based script, and every explanation method below is rewired around
# it. Because there is only one such layer, the old `n` (last-n-layers)
# parameter no longer has a meaning and is fixed at 1.
#
# Required packages: open_clip_torch, timm, transformers (the SigLIP
# tokenizer is HF-based), torch, torchvision, opencv-python, matplotlib.
# ---------------------------------------------------------------------------

model_name = "ViT-B-16-SigLIP-256"
pretrained_tag = "webli"  # the only pretrained tag open_clip ships for this arch/resolution
device = "cuda" if torch.cuda.is_available() else "cpu"

clipmodel, preprocess = create_model_from_pretrained(model_name, pretrained=pretrained_tag, device=device)
tokenizer = get_tokenizer(model_name)
clipmodel.eval()

visual = clipmodel.visual            # open_clip TimmModel wrapper
trunk = visual.trunk                 # timm.models.vision_transformer.VisionTransformer
attn_pool = trunk.attn_pool          # AttentionPoolLatent: SigLIP's MAP pooling head

clip_inres = visual.image_size[0]            # 256
clip_ksize = trunk.patch_embed.patch_size    # (16, 16)

# SigLIP/timm preprocessing uses "inception-style" normalization, not CLIP's
# OpenAI stats. This MUST match what `preprocess` (above) does internally,
# since this constant is reused for the dense (aspect-ratio-preserving) path.
_transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
label_dict = {
    "ham": ["Basal cell carcinoma", "Benign keratosis", "Dermatofibroma", "Melanocytic nevi", "Melanoma", "Vascular skin lesions"],
    "pad": ["Actinic Keratosis", "Basal Cell Carcinoma", "Malignant Melanoma", "Melanocytic Nevus", "Squamous Cell Carcinoma", "Seborrheic Keratosis"]
}


def cap_long_side(img, max_side=896):
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    return img


def imgprocess(img, patch_size=clip_ksize, scale_factor=1):
    """Resize to a multiple of the patch size while *preserving aspect
    ratio* (unlike the model's own preprocessing, which squashes to a
    square). This is what makes the dense localization maps meaningful."""
    w, h = img.size
    ph, pw = patch_size
    nw = int(w * scale_factor / pw + 0.5) * pw
    nh = int(h * scale_factor / ph + 0.5) * ph

    ResizeOp = Resize((nh, nw), interpolation=InterpolationMode.BICUBIC)
    img = ResizeOp(img).convert("RGB")
    return _transform(img)


def visualize(map, raw_image, resize):
    image = np.asarray(raw_image.copy())
    map = resize(map.unsqueeze(0))[0].cpu().numpy()
    color = cv2.applyColorMap((map * 255).astype(np.uint8), cv2.COLORMAP_JET)  # cv2 to plt
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    c_ret = np.clip(image * (1 - 0.5) + color * 0.5, 0, 255).astype(np.uint8)
    return c_ret


def attention_layer(q, k, v, num_heads=1):
    "Compute 'Scaled Dot Product Attention', merging all heads into one for a cleaner explanation map."
    tgt_len, bsz, embed_dim = q.shape
    head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_heads = torch.bmm(attn_output_weights, v)
    assert list(attn_output_heads.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output_heads.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, -1)
    attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
    return attn_output, None  # attn_output_weights


def siglip_encode_dense(x):
    """Dense re-implementation of the SigLIP visual forward pass, exposing
    the q/k/v/attn_output of the MAP pooling head so gradients can be
    pulled out for Grad-ECLIP / Grad-CAM, the same way the original
    `clip_encode_dense` exposed the CLS self-attention internals.

    Returns the pooled (whole-image) embedding, the pre-pooling patch
    feature map (the GradCAM target), and lists (length 1, since SigLIP has
    only one pooling layer) of q, k, v, attention weights and attn_output,
    kept as lists so grad_eclip's multi-layer-aggregation loop is unchanged.
    """
    feah, feaw = None, None

    x = trunk.patch_embed.proj(x)            # [B, C, H', W']
    feah, feaw = x.shape[-2:]
    x = x.flatten(2).transpose(1, 2)         # [B, N, C]

    # Scale the position embedding to the actual (possibly non-square)
    # patch grid, exactly like the original script did for CLIP -- except
    # SigLIP's pos_embed has no leading CLS-token row to split off.
    pos_h, pos_w = trunk.patch_embed.grid_size
    pos_embedding = trunk.pos_embed  # [1, pos_h*pos_w, C]
    assert pos_embedding.size(1) == (pos_h * pos_w), \
        f"the size of pos_embedding ({pos_embedding.size(1)}) does not match resolution shape pos_h ({pos_h}) * pos_w ({pos_w})"
    pos_embedding = pos_embedding.reshape(1, pos_h, pos_w, -1).permute(0, 3, 1, 2)
    pos_embedding = F.interpolate(pos_embedding, size=(feah, feaw), mode='bicubic', align_corners=False)
    pos_embedding = pos_embedding.reshape(1, pos_embedding.shape[1], -1).permute(0, 2, 1)  # [1, N, C]
    x = x + pos_embedding
    x = trunk.patch_drop(x)
    x = trunk.norm_pre(x)

    x = trunk.blocks(x)
    x = trunk.norm(x)                         # [B, N, C] -- pre-pooling patch features

    x = x.permute(1, 0, 2)                    # NLD -> LND, matching the original script's convention
    last_feat = x                             # the actual tensor consumed below -> valid autograd target

    bsz = x.shape[1]
    q_latent = attn_pool.latent.expand(bsz, -1, -1).permute(1, 0, 2)  # [1, B, C] -- the learned pooling query
    q = attn_pool.q(q_latent)                  # [1, B, C]
    kv = attn_pool.kv(x)                       # [N, B, 2C]  (nn.Linear acts on the last dim only)
    k, v = kv.chunk(2, dim=-1)                 # each [N, B, C]  -- all patches, no prefix token to drop

    attn_output, attn = attention_layer(q, k, v, num_heads=1)  # [1, B, C]

    pooled = attn_pool.proj(attn_output)
    pooled = attn_pool.proj_drop(pooled)
    pooled = pooled + attn_pool.mlp(attn_pool.norm(pooled))
    pooled = pooled[0]                          # [B, C] final image embedding

    return pooled, last_feat, [v], [q], [k], [attn], [attn_output], (feah, feaw)


def patch_projection(v):
    """Approximate per-patch embedding in the same space as the pooled
    image embedding, by pushing each patch's value vector through the same
    proj+MLP pathway the pooled output goes through (as if that single
    patch alone were attended). SigLIP has no native per-token projection
    the way CLIP's ln_post+proj did, so this mirrors the spirit of the
    original script's dense `p_final` computation rather than being an
    architectural ground truth."""
    x = attn_pool.proj(v)
    x = attn_pool.proj_drop(x)
    x = x + attn_pool.mlp(attn_pool.norm(x))
    return x


def sim_qk(q, k):
    q_probe = F.normalize(q[:, 0, :], dim=-1)   # [1, C] -- the single pooling-latent query
    k_patch = F.normalize(k[:, 0, :], dim=-1)   # [N, C] -- all patch keys (no CLS to drop)

    cosine_qk = (q_probe * k_patch).sum(-1)
    cosine_qk_max = cosine_qk.max(dim=-1, keepdim=True)[0]
    cosine_qk_min = cosine_qk.min(dim=-1, keepdim=True)[0]
    cosine_qk = (cosine_qk - cosine_qk_min) / (cosine_qk_max - cosine_qk_min)
    return cosine_qk


def grad_eclip(c, qs, ks, vs, attn_outputs, map_size):
    ## gradient on the pooling attention's output
    tmp_maps = []
    for q, k, v, attn_output in zip(qs, ks, vs, attn_outputs):
        grad = torch.autograd.grad(
            c,
            attn_output,
            retain_graph=True)[0]

        grad_cls = grad[:, 0, :]        # [1, C] -- gradient on the pooled-query's output
        v_patch = v[:, 0, :]            # [N, C] -- all patch values (no CLS to drop)
        cosine_qk = sim_qk(q, k).reshape(-1)
        tmp_maps.append((grad_cls * v_patch * cosine_qk[:, None]).sum(-1))

    emap = F.relu_(torch.stack(tmp_maps, dim=0)).sum(0)
    return emap.reshape(*map_size)


def self_attn(attns, map_size):
    """Kept for parity with the original script. Note: just like in the
    original, `attention_layer` always returns None for the attention
    weights (the real softmax map is discarded after merging heads), so
    this is unused/inert in main() in both versions unless
    `attention_layer` is changed to also return `attn_output_weights`."""
    attn_patch = attns[-1][0, :, :].reshape(*map_size)
    print("[attn of pooling query on patches]:", attn_patch.shape)
    return attn_patch


def grad_cam(c, feat, map_size):
    ## GRAD-CAM: use the patch feature map that feeds into the pooling head
    grad = torch.autograd.grad(
        c,
        feat,
        retain_graph=True)[0]
    grad_weight = grad.mean(0, keepdim=True)
    grad_cam = F.relu_((grad_weight * feat).sum(-1))
    grad_cam = grad_cam.reshape(*map_size)  # no CLS token to drop for SigLIP
    return grad_cam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Name of the folder containing jpg images")
    args = parser.parse_args()

    if args.folder not in ("ham", "pad"):
        print("Dataset not implemented")
        return

    print("[siglip resolution]:", clip_inres)
    print("[siglip patch size]:", clip_ksize)

    print()
    jpg_files = sorted(f for f in os.listdir(f"../{args.folder}_images") if f.lower().endswith((".jpg", ".png")))

    texts = label_dict[args.folder]

    for img_idx, fname in enumerate(jpg_files):
        img_path = os.path.join(f"../{args.folder}_images", fname)
        img = Image.open(img_path).convert("RGB")
        print("=" * 30)
        print(f"Loaded {fname}, size={img.size}")

        img = cap_long_side(img)

        # preprocess image and text
        img_preprocessed = preprocess(img).to(device).unsqueeze(0)
        text_processed = tokenizer(texts).to(device)
        # extract text feature
        text_embedding = clipmodel.encode_text(text_processed)
        text_embedding = F.normalize(text_embedding, dim=-1)
        print("[text embedding]:", text_embedding.shape)

        ori_img_embedding = clipmodel.encode_image(img_preprocessed)
        ori_img_embedding = F.normalize(ori_img_embedding, dim=-1)
        print("[image embedding]:", ori_img_embedding.shape)

        cosine = (ori_img_embedding @ text_embedding.T)
        # SigLIP's actual classification score is a per-class sigmoid, not a
        # softmax-over-cosine -- show it too, for the true class probabilities.
        siglip_probs = torch.sigmoid(cosine * clipmodel.logit_scale.exp() + clipmodel.logit_bias)
        print("[cosine]:", cosine)
        print("[siglip probs]:", siglip_probs)

        img_preprocessed_k = imgprocess(img).to(device).unsqueeze(0)
        #
        # pooled, last_feat, vs, qs, ks, attns, atten_outs, (feah, feaw)
        outputs, last_feat, vs, qs, ks, attns, atten_outs, map_size = siglip_encode_dense(img_preprocessed_k)
        img_embedding = F.normalize(outputs, dim=-1)
        print("[image embedding]:", img_embedding.shape)
        cosine = (img_embedding @ text_embedding.T)[0]
        print("cosine:", cosine)

        # similarity between text prompt and patch features (approximate dense readout)
        p_final = F.normalize(patch_projection(vs[0]), dim=-1)
        cosine_p = (p_final[:, 0, :] @ text_embedding.T).transpose(1, 0).reshape(-1, *map_size)
        print("[position similarity (cosine p)]:", cosine_p.shape)

        # NOTE: we use raw cosine similarity (not the sigmoid-scaled SigLIP
        # logit) as the gradient target for the explanation maps below.
        # SigLIP's logit_bias is initialized very negative (~-10), so the
        # sigmoid saturates and its gradient vanishes for any class the
        # model isn't already confident about -- using cosine avoids that
        # saturation and keeps the explanation gradients informative for
        # every class, not just the predicted one.
        grad_emaps = []
        grad_cams = []
        for i, c in enumerate(cosine):
            grad_emaps.append(grad_eclip(c, qs, ks, vs, atten_outs, map_size))
            grad_cams.append(grad_cam(c, last_feat, map_size))

        print(texts)
        h, w = img.size
        resize = T.Resize((w, h))
        fig, axs = plt.subplots(ncols=len(cosine), nrows=2, figsize=(15, 6))
        for i, ax in enumerate(axs.T):
            tmp = grad_emaps[i].clone()
            tmp -= tmp.min()
            tmp /= tmp.max()
            c_ret = visualize(tmp.detach().cpu(), img, resize)
            ax[0].axis('off')
            ax[0].imshow(c_ret)

            tmp = grad_cams[i].clone()
            tmp -= tmp.min()
            tmp /= tmp.max()
            c_ret = visualize(tmp.detach().cpu(), img, resize)
            ax[1].axis('off')
            ax[1].imshow(c_ret)

        fig.savefig(f'maps/{args.folder}/{model_name.replace("/", "")}_heatmap_{img_idx}.png', dpi=600, bbox_inches='tight')

        plt.close(fig)

        del outputs, last_feat, vs, qs, ks, attns, atten_outs
        del img_preprocessed, img_preprocessed_k, text_embedding, ori_img_embedding
        del grad_emaps, grad_cams, cosine, cosine_p, p_final, text_processed, siglip_probs
        if device == "cuda":
            torch.cuda.empty_cache()

    print('-- DONE --')


if __name__ == "__main__":
    main()