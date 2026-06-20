import argparse
import math
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from open_clip import create_model_from_pretrained, get_tokenizer
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

model_name = "hf-hub:redlessone/DermLIP_ViT-B-16"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model loading (replaces clip.load) ---
# create_model_from_pretrained downloads the HF-hub checkpoint and returns
# the model in eval mode together with its inference preprocessing transform.
clipmodel, preprocess = create_model_from_pretrained(model_name)
clipmodel = clipmodel.to(device).eval()

# OpenCLIP uses a separate tokenizer object instead of clip.tokenize()
tokenizer = get_tokenizer(model_name)

# OpenCLIP's VisionTransformer exposes image_size as a (H, W) tuple,
# whereas OpenAI CLIP stored it as a plain int in input_resolution.
clip_inres = clipmodel.visual.image_size[0]   # e.g. 224 for ViT-B/16
clip_ksize = clipmodel.visual.conv1.kernel_size  # still an nn.Conv2d → tuple

# OpenCLIP defaults to fp32; capture the actual dtype so the manual forward
# pass in clip_encode_dense always matches the model's parameter dtype.
clip_dtype = next(clipmodel.parameters()).dtype


# ---------------------------------------------------------------------------
# Helper: extract Q, K, V projections from an attention module.
#
# OpenCLIP ships two different attention layouts depending on version / config:
#   1. nn.MultiheadAttention  – single fused in_proj_weight  (shape 3D×D)
#   2. Custom Attention class – separate q_proj / k_proj / v_proj nn.Linears
#
# Both expose out_proj as an nn.Linear, so that path is unconditional.
# ---------------------------------------------------------------------------
def _get_qkv(attn_module, x):
    linear = torch._C._nn.linear
    if hasattr(attn_module, 'in_proj_weight') and attn_module.in_proj_weight is not None:
        # Layout 1: nn.MultiheadAttention (or OpenAI-style fused projection)
        q, k, v = linear(x, attn_module.in_proj_weight,
                          attn_module.in_proj_bias).chunk(3, dim=-1)
    elif hasattr(attn_module, 'q_proj'):
        # Layout 2: separate projection linears (OpenCLIP custom Attention)
        # nn.Linear.bias is None when bias=False, which F.linear handles fine.
        q = linear(x, attn_module.q_proj.weight, attn_module.q_proj.bias)
        k = linear(x, attn_module.k_proj.weight, attn_module.k_proj.bias)
        v = linear(x, attn_module.v_proj.weight, attn_module.v_proj.bias)
    else:
        raise AttributeError(
            f"Cannot find QKV projection weights in attention module of type "
            f"{type(attn_module)}. Expected either 'in_proj_weight' "
            "(nn.MultiheadAttention) or 'q_proj'/'k_proj'/'v_proj'."
        )
    return q, k, v


def cap_long_side(img, max_side=896):
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    return img

def imgprocess(img, patch_size=[16, 16], scale_factor=1):
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
    "Compute 'Scaled Dot Product Attention'"
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
    return attn_output, None # attn_output_weights

    
def clip_encode_dense(x, n):
    vision_width = clipmodel.visual.transformer.width
    vision_heads = vision_width // 64
    print("[vision_width and vision_heads]:", vision_width, vision_heads)
    
    # Cast input to the model's parameter dtype.
    # OpenAI CLIP ran in fp16 (hence the original x.half()); OpenCLIP defaults
    # to fp32.  Using clip_dtype keeps this correct in both cases.
    x = x.to(clip_dtype)
    x = clipmodel.visual.conv1(x)
    feah, feaw = x.shape[-2:]

    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    class_embedding = clipmodel.visual.class_embedding.to(x.dtype)
    x = torch.cat([class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1]).to(x), x], dim=1)

    ## scale position embedding as the image w-h ratio
    pos_embedding = clipmodel.visual.positional_embedding.to(x.dtype)
    tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
    pos_h = clip_inres // clip_ksize[0]
    pos_w = clip_inres // clip_ksize[1]
    assert img_pos.size(0) == (pos_h * pos_w), (
        f"the size of pos_embedding ({img_pos.size(0)}) does not match "
        f"resolution shape pos_h ({pos_h}) * pos_w ({pos_w})"
    )
    img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
    print("[POS shape]:", img_pos.shape, (feah, feaw))
    img_pos = torch.nn.functional.interpolate(img_pos, size=(feah, feaw), mode='bicubic', align_corners=False)
    img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)
    pos_embedding = torch.cat((tok_pos[None, ...], img_pos), dim=1)
    x = x + pos_embedding
    x = clipmodel.visual.ln_pre(x)
    
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = torch.nn.Sequential(*clipmodel.visual.transformer.resblocks[:-n])(x)

    attns = []
    atten_outs = []
    vs = []
    qs = []
    ks = []
    linear = torch._C._nn.linear
    for TR in clipmodel.visual.transformer.resblocks[-n:]:
        x_in = x
        x = TR.ln_1(x_in)

        # Use helper to support both nn.MultiheadAttention and custom layouts
        q, k, v = _get_qkv(TR.attn, x)
        attn_output, attn = attention_layer(q, k, v, 1)  # vision_heads=1
        attns.append(attn)
        atten_outs.append(attn_output)
        vs.append(v)
        qs.append(q)
        ks.append(k)

        x_after_attn = linear(attn_output, TR.attn.out_proj.weight, TR.attn.out_proj.bias)
        # Apply layer scale on the attention branch if present.
        # Newer OpenCLIP blocks include ls_1 / ls_2 (nn.Identity when
        # ls_init_value=None, so this is a no-op for standard ViT-B/16).
        x_after_attn = TR.ls_1(x_after_attn) if hasattr(TR, 'ls_1') else x_after_attn
        x = x_after_attn + x_in

        mlp_out = TR.mlp(TR.ln_2(x))
        # Apply layer scale on the MLP branch if present.
        mlp_out = TR.ls_2(mlp_out) if hasattr(TR, 'ls_2') else mlp_out
        x = x + mlp_out

    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clipmodel.visual.ln_post(x)
    x = x @ clipmodel.visual.proj
    return x, x_in, vs, qs, ks, attns, atten_outs, (feah, feaw)

def sim_qk(q, k):
    q_cls = F.normalize(q[:1,0,:], dim=-1) 
    k_patch = F.normalize(k[1:,0,:], dim=-1)

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

        grad_cls = grad[:1,0,:]
        v_patch = v[1:,0,:]
        cosine_qk = sim_qk(q, k).reshape(-1)
        tmp_maps.append((grad_cls * v_patch * cosine_qk[:,None]).sum(-1)) 

    emap = F.relu_(torch.stack(tmp_maps, dim=0)).sum(0)
    return emap.reshape(*map_size)

def self_attn(attns, map_size):
    attn_patch = attns[-1][0,:1,1:].reshape(*map_size)
    print("[attn of cls token on lastv]:", attn_patch.shape)
    return attn_patch

def grad_cam(c, feat, map_size):
    ## GRAD-CAM: use the feature outputs of the final attention layer
    grad = torch.autograd.grad(
        c,
        feat,
        retain_graph=True)[0]
    grad_weight = grad.mean(0, keepdim=True)
    grad_cam = F.relu_((grad_weight * feat).sum(-1))
    grad_cam = grad_cam[1:].reshape(*map_size)
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
        # preprocess comes from create_model_from_pretrained (same CLIP normalisation)
        img_preprocessed = preprocess(img).to(device).unsqueeze(0)
        # tokenizer replaces clip.tokenize(); output shape and semantics are identical
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
        # x, vs, qs, ks, attns, atten_outs, (feah, feaw)
        outputs, last_feat, vs, qs, ks, attns, atten_outs, map_size = clip_encode_dense(img_preprocessed_k, n=1)
        img_embedding = F.normalize(outputs[:,0], dim=-1)
        print("[image embedding]:", img_embedding.shape)
        cosine = (img_embedding @ text_embedding.T)[0]
        print("cosine:", cosine)

        # similarity between text prompt and patch features
        p_final = F.normalize(outputs[:,1:], dim=-1)
        cosine_p = (p_final @ text_embedding.T)[0].transpose(1,0).reshape(-1, *map_size)
        print("[position similarity (cosine p)]:", cosine_p.shape)

        grad_emaps = []
        grad_cams = []
        for i, c in enumerate(cosine):
            grad_emaps.append(grad_eclip(c, qs, ks, vs, atten_outs, map_size))
            grad_cams.append(grad_cam(c, last_feat, map_size))

        print(texts)
        h, w = img.size
        resize = T.Resize((w,h))
        fig, axs = plt.subplots(ncols=len(cosine), nrows=2, figsize=(30, 12))
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

        # Use the last path component as the filename slug to avoid colons / slashes
        model_slug = model_name.split("/")[-1]  # "DermLIP_ViT-B-16"
        fig.savefig(f'maps/{args.folder}/{model_slug}_heatmap_{img_idx}.png', dpi=600, bbox_inches='tight')

        plt.close(fig)
        
        del outputs, last_feat, vs, qs, ks, attns, atten_outs
        del img_preprocessed, img_preprocessed_k, text_embedding, ori_img_embedding
        del grad_emaps, grad_cams, cosine, cosine_p, p_final, text_processed
        torch.cuda.empty_cache()
    
    print('-- DONE --')

if __name__ == "__main__":
    main()