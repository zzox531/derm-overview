import argparse
import math
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
import open_clip
import cv2
import numpy as np
import os
from urllib.request import urlopen
import matplotlib.pyplot as plt
from PIL import Image

_transform = Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
label_dict = {
    "ham": ["Basal cell carcinoma", "Benign keratosis", "Dermatofibroma", "Melanocytic nevi", "Melanoma", "Vascular skin lesions"],
    "pad": ["Actinic Keratosis", "Basal Cell Carcinoma", "Malignant Melanoma", "Melanocytic Nevus", "Squamous Cell Carcinoma", "Seborrheic Keratosis"]
}

# ----------------------------------------------------------------------------------
# CoCa ViT-B-32 (open_clip) setup.
#
# Key differences from OpenAI CLIP (clip.load) that this script accounts for:
#   1. open_clip's CoCa visual tower (open_clip.transformer.VisionTransformer) runs
#      its transformer BATCH-FIRST: tensors are (N, L, D), not CLIP's (L, N, D).
#   2. CoCa's image tower has an AttentionalPooler ("attn_pool") after the last
#      resblock. The contrastive image embedding ("pooled") and the 256 caption
#      tokens ("tokens") it returns both come from *learned pooling queries*, NOT
#      from the original 49 (7x7) spatial patches -- so they cannot be reshaped to
#      a spatial map directly.
#   3. To get spatially-meaningful localization maps (Grad-ECLIP / Grad-CAM /
#      self-attention), we instead hook into the PRE-pool transformer sequence
#      (1 CLS token + 49 patch tokens for ViT-B-32 @ 224px) and backprop the
#      gradient of the final contrastive similarity THROUGH the attentional
#      pooler back to that pre-pool sequence. This works because attn_pool is a
#      standard differentiable cross-attention module.
#   4. CoCa's resblocks use nn.MultiheadAttention(batch_first=True) rather than
#      CLIP's manual q/k/v projections, but the in_proj_weight/in_proj_bias/
#      out_proj parameters live in the same place (TR.attn.in_proj_weight, etc.),
#      so we can still manually recompute q, k, v with autograd-visible ops.
# ----------------------------------------------------------------------------------

model_name = "coca_ViT-B-32"
pretrained_tag = "laion2b_s13b_b90k"
device = "cuda" if torch.cuda.is_available() else "cpu"

clipmodel, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag)
clipmodel = clipmodel.to(device).eval()
tokenizer = open_clip.get_tokenizer(model_name)

clip_inres = clipmodel.visual.image_size[0]
clip_ksize = clipmodel.visual.patch_size  # (32, 32) for ViT-B-32

def cap_long_side(img, max_side=896):
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    return img

def imgprocess(img, patch_size=[32, 32], scale_factor=1):
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
    color = cv2.applyColorMap((map*255).astype(np.uint8), cv2.COLORMAP_JET) # cv2 to plt
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    c_ret = np.clip(image * (1 - 0.5) + color * 0.5, 0, 255).astype(np.uint8)
    return c_ret

def attention_layer(q, k, v, num_heads=1):
    """Compute 'Scaled Dot Product Attention'.

    NOTE: unlike the CLIP version, q/k/v here are BATCH-FIRST: (bsz, tgt_len, embed_dim).
    """
    bsz, tgt_len, embed_dim = q.shape
    head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5
    q = q * scaling

    q = q.contiguous().view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)  # bsz, heads, tgt_len, head_dim
    k = k.contiguous().view(bsz, -1, num_heads, head_dim).transpose(1, 2)
    v = v.contiguous().view(bsz, -1, num_heads, head_dim).transpose(1, 2)

    q = q.reshape(bsz * num_heads, tgt_len, head_dim)
    k = k.reshape(bsz * num_heads, -1, head_dim)
    v = v.reshape(bsz * num_heads, -1, head_dim)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_heads = torch.bmm(attn_output_weights, v)
    assert list(attn_output_heads.size()) == [bsz * num_heads, tgt_len, head_dim]

    attn_output = attn_output_heads.view(bsz, num_heads, tgt_len, head_dim).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, -1)
    attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
    return attn_output, None  # attn_output_weights

def clip_encode_dense(x, n):
    visual = clipmodel.visual
    vision_width = visual.transformer.width
    vision_heads = visual.transformer.resblocks[0].attn.num_heads
    print("[vision_width and vision_heads]:", vision_width, vision_heads)

    # modified from open_clip VisionTransformer._embeds, kept batch-first (N, L, D)
    x = x.half() if next(visual.parameters()).dtype == torch.float16 else x
    x = visual.conv1(x)
    feah, feaw = x.shape[-2:]

    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)  # N, L, D  (batch-first, no permute to LND like CLIP)
    class_embedding = visual.class_embedding.to(x.dtype)
    x = torch.cat([class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1).to(x), x], dim=1)

    ## scale position embedding as the image w-h ratio
    pos_embedding = visual.positional_embedding.to(x.dtype)
    tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
    pos_h = clip_inres // clip_ksize[0]
    pos_w = clip_inres // clip_ksize[1]
    assert img_pos.size(0) == (pos_h * pos_w), f"the size of pos_embedding ({img_pos.size(0)}) does not match resolution shape pos_h ({pos_h}) * pos_w ({pos_w})"
    img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
    print("[POS shape]:", img_pos.shape, (feah, feaw))
    img_pos = torch.nn.functional.interpolate(img_pos, size=(feah, feaw), mode='bicubic', align_corners=False)
    img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)
    pos_embedding = torch.cat((tok_pos[None, ...], img_pos), dim=1)
    x = x + pos_embedding
    x = visual.ln_pre(x)

    # x stays (N, L, D) -- batch-first, no LND permute needed for open_clip's Transformer
    x = torch.nn.Sequential(*visual.transformer.resblocks[:-n])(x)

    attns = []
    atten_outs = []
    vs = []
    qs = []
    ks = []
    for TR in visual.transformer.resblocks[-n:]:
        x_in = x
        x = TR.ln_1(x_in)
        linear = torch._C._nn.linear
        # TR.attn is nn.MultiheadAttention; in_proj_weight/in_proj_bias/out_proj are
        # the same parameter names/shapes as CLIP's custom attention, but operate
        # batch-first here: q,k,v are (bsz, seq_len, embed_dim).
        q, k, v = linear(x, TR.attn.in_proj_weight, TR.attn.in_proj_bias).chunk(3, dim=-1)
        attn_output, attn = attention_layer(q, k, v, 1)  # collapse all heads into 1, like the CLIP version
        attns.append(attn)
        atten_outs.append(attn_output)
        vs.append(v)
        qs.append(q)
        ks.append(k)

        x_after_attn = linear(attn_output, TR.attn.out_proj.weight, TR.attn.out_proj.bias)
        x = x_after_attn + x_in
        x = x + TR.mlp(TR.ln_2(x))

    # x is the pre-pool sequence (N, L, D) = (N, 1 + feah*feaw, D)
    last_feat_pre_pool = x

    # Run CoCa's attentional pooler + ln_post + global pool + proj, exactly as
    # VisionTransformer._pool / forward do, so `outputs` matches what
    # clipmodel.encode_image(...) would have produced (the contrastive embedding).
    pooled_seq = visual.attn_pool(x)       # (N, 256, D) cross-attention pooling queries
    pooled_seq = visual.ln_post(pooled_seq)
    pooled, cap_tokens = visual._global_pool(pooled_seq)  # pool_type='tok' -> pooled = pooled_seq[:,0]
    if visual.proj is not None:
        pooled = pooled @ visual.proj

    # `outputs` plays the same role as CLIP's `x` in the original script: position 0
    # is the pooled/contrastive embedding. We do NOT have per-spatial-patch final
    # embeddings post-pool (CoCa's pooling destroys that spatial correspondence), so
    # downstream code must use last_feat_pre_pool (pre-pool CLS+patches) for any
    # spatial localization, and `pooled` only for the global cosine similarity.
    outputs = pooled.unsqueeze(1)  # (N, 1, D) -- kept 3D to mirror outputs[:,0] usage below

    return outputs, last_feat_pre_pool, vs, qs, ks, attns, atten_outs, (feah, feaw)

def sim_qk(q, k):
    # q, k are batch-first here: (bsz, seq_len, embed_dim); we operate on batch index 0
    q_cls = F.normalize(q[0, :1, :], dim=-1)
    k_patch = F.normalize(k[0, 1:, :], dim=-1)

    cosine_qk = (q_cls * k_patch).sum(-1)
    cosine_qk_max = cosine_qk.max(dim=-1, keepdim=True)[0]
    cosine_qk_min = cosine_qk.min(dim=-1, keepdim=True)[0]
    cosine_qk = (cosine_qk-cosine_qk_min) / (cosine_qk_max-cosine_qk_min)
    return cosine_qk

def grad_eclip(c, qs, ks, vs, attn_outputs, map_size):
    ## gradient on last attention output
    tmp_maps = []
    for q, k, v, attn_output in zip(qs, ks, vs, attn_outputs):
        grad = torch.autograd.grad(
            c,
            attn_output,
            retain_graph=True)[0]

        # batch-first: grad/v are (bsz, seq_len, embed_dim); take batch 0
        grad_cls = grad[0, :1, :]
        v_patch = v[0, 1:, :]
        cosine_qk = sim_qk(q, k).reshape(-1)
        tmp_maps.append((grad_cls * v_patch * cosine_qk[:, None]).sum(-1))

    emap = F.relu_(torch.stack(tmp_maps, dim=0)).sum(0)
    return emap.reshape(*map_size)

def self_attn(attns, map_size):
    attn_patch = attns[-1][0, :1, 1:].reshape(*map_size)
    print("[attn of cls token on lastv]:", attn_patch.shape)
    return attn_patch

def grad_cam(c, feat, map_size):
    ## GRAD-CAM: use the (pre-pool) feature outputs of the final attention layer
    ## feat is batch-first: (bsz, seq_len, embed_dim)
    grad = torch.autograd.grad(
        c,
        feat,
        retain_graph=True)[0]
    grad_weight = grad.mean(1, keepdim=True)  # average over sequence dim (batch-first -> dim=1)
    grad_cam = F.relu_((grad_weight * feat).sum(-1))
    grad_cam = grad_cam[0, 1:].reshape(*map_size)
    return grad_cam

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Name of the folder containing jpg images")
    args = parser.parse_args()

    if args.folder not in ("ham", "pad"):
        print("Dataset not implemented")
        return

    print("[clip resolution]:", clip_inres)
    print("[clip kernel size]:", clip_ksize)

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
        print("[cosine]:", cosine)

        img_preprocessed_k = imgprocess(img).to(device).unsqueeze(0)
        #
        # outputs, last_feat, vs, qs, ks, attns, atten_outs, (feah, feaw)
        outputs, last_feat, vs, qs, ks, attns, atten_outs, map_size = clip_encode_dense(img_preprocessed_k, n=1)
        img_embedding = F.normalize(outputs[:, 0], dim=-1)
        print("[image embedding]:", img_embedding.shape)
        cosine = (img_embedding @ text_embedding.T)[0]
        print("cosine:", cosine)

        # NOTE: CoCa's attentional pooler destroys the 1:1 token<->spatial-patch
        # correspondence (pooled output is 256 learned-query tokens, not a 7x7 grid),
        # so a per-position "patch vs text" similarity map (cosine_p in the CLIP
        # version) cannot be computed the same way here. We skip it; Grad-ECLIP and
        # Grad-CAM below still recover spatial localization via gradients into the
        # pre-pool patch tokens, which is the more reliable signal anyway.

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

        os.makedirs(f'maps/{args.folder}', exist_ok=True)
        fig.savefig(f'maps/{args.folder}/{model_name.replace("/", "")}_heatmap_{img_idx}.png', dpi=600, bbox_inches='tight')

        plt.close(fig)

        del outputs, last_feat, vs, qs, ks, attns, atten_outs
        del img_preprocessed, img_preprocessed_k, text_embedding, ori_img_embedding
        del grad_emaps, grad_cams, cosine, text_processed
        if device == "cuda":
            torch.cuda.empty_cache()

    print('-- DONE --')

if __name__ == "__main__":
    main()