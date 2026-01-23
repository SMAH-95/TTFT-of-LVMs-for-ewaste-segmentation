import os, time, math, random, numpy as np
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import shutil
import json
from scipy.ndimage import label as connected_components, binary_dilation, binary_erosion
from activeft.sift import Retriever
from activeft.acquisition_functions.itl import ITL
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# config
GALLERY_DIR = "./Dataset/candidate_images"
QUERY_DIR   = "./Dataset/query_images"
CKPT_PATH   = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_VARIANT = "large"     # "tiny" | "small" | "base" | "large"
OUT_DIR = Path("./Results/point_prompt_seg")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_OUT_DIR = OUT_DIR / "masks"
MASK_OUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOPN = 25
BATCH_SIZE = 8
K_PRESELECT = 300
ALPHA = None
NUM_WORKERS = 2
PIN_MEMORY = (DEVICE == "cuda")
MASK_SUFFIX = "_mask.png"
POINTS_DIR = Path("./Prompt/point_prompt")
POINTS_HAVE_CLASS_ID = True
POINTS_CLASS_OFFSET  = 0            
POINTS_COORDS_NORMALIZED = False    
SIM_THRESHOLD       = 0.25
RESOLVE_TIES_HIGHER = True
MIN_PROPOSAL_PIXELS = 16 
TTFT_UPDATES      = 5
TTFT_ACCUM_STEPS  = 1
TTFT_LR           = 1e-5
TTFT_WD           = 1e-6
TTFT_CLIP_NORM    = 1.0
TTFT_MU_PROX      = 0.0
MAX_EMPTY_TRIES   = 50
ITL_NOISE_STD = 0.5
ITL_MINI_BATCH = 1000
ITL_SUBSAMPLED_TARGET_FRAC = 1.0
ITL_MAX_TARGET_SIZE = None
ITL_TARGET_PATCHES = 64 


CLASS_NAMES = ["pv_panel_cell","pv_panel_frame","pv_panel_support","bulb_ballast","bulb_core","bulb_glass","cable_conductor","cable_insulation","duct_diffuser","duct_insulation", "duct_trunk","outlet_cable","outlet_mounting_strap","outlet_wallplate",]
NUM_CLASSES = len(CLASS_NAMES)
PALETTE_HEX = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33","#a65628","#f781bf","#999999","#66c2a5","#e7298a","#1b9e77","#d95f02","#7570b3"]

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

CLASS_RGB = [_hex_to_rgb(h) for h in PALETTE_HEX]
FULL_RGB_PALETTE = [(0,0,0)] + CLASS_RGB  

def _pil_palette_256(rgb_list):
    flat = []
    for i in range(256):
        if i < len(rgb_list): flat += list(rgb_list[i])
        else: flat += [0,0,0]
    return flat

PIL_PALETTE_256 = _pil_palette_256(FULL_RGB_PALETTE)

def rgb_mask_to_class_indices(rgb_mask: np.ndarray) -> np.ndarray:
    h, w, _ = rgb_mask.shape
    pal = np.asarray(FULL_RGB_PALETTE, dtype=np.uint8)  
    flat = rgb_mask.reshape(-1,3).astype(np.uint8)     
    eq = (flat[:,None,:] == pal[None,:,:]).all(axis=2)  
    has = eq.any(axis=1)
    idx = eq.argmax(axis=1)                             
    idx[~has] = 0
    return idx.reshape(h,w).astype(np.uint8)

def save_color_mask_idx(mask_idx_u8: np.ndarray, out_path: Path):
    im = Image.fromarray(mask_idx_u8, mode="P")
    im.putpalette(PIL_PALETTE_256)
    im.save(out_path)

def _to_py(x):
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x

def set_seeds(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seeds(42)

def _resolve_config_candidates(version: str, variant: str):
    size_map = {"tiny": "t", "small": "s", "base": "b", "large": "l"}
    s = size_map[variant]
    if version == "sam2.1":
        base = f"sam2.1_hiera_{s}.yaml"
        return [f"configs/sam2.1/{base}", f"sam2.1/{base}", base]
    else:
        base = f"sam2_hiera_{s}.yaml"
        return [f"configs/sam2/{base}", f"sam2/{base}", base]

ckpt_path = Path(CKPT_PATH)
if not ckpt_path.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

last_err = None
for cfg_name in _resolve_config_candidates(SAM2_VERSION, SAM2_VARIANT):
    try:
        sam2_model = build_sam2(cfg_name, str(ckpt_path), device=DEVICE)
        break
    except Exception as e:
        last_err = e
else:
    raise RuntimeError(
        f"Could not load a SAM2 config for version={SAM2_VERSION}, variant={SAM2_VARIANT}.\n"
        f"Tried: {_resolve_config_candidates(SAM2_VERSION, SAM2_VARIANT)}\n"
        f"Last error: {last_err}"
    )

predictor = SAM2ImagePredictor(sam2_model)

prompt_enc = getattr(sam2_model, "sam_prompt_encoder", getattr(sam2_model, "prompt_encoder", None))
mask_dec   = getattr(sam2_model, "sam_mask_decoder",   getattr(sam2_model, "mask_decoder",   None))
if prompt_enc is None or mask_dec is None:
    raise AttributeError("SAM2 model does not expose prompt/decoder modules.")

dense_pe = prompt_enc.get_dense_pe() if hasattr(prompt_enc, "get_dense_pe") else None

try:
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(CLIP_DEVICE).eval()
    def _clip_encode(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            z = clip_model.encode_image(img_batch.to(CLIP_DEVICE))
        return F.normalize(z.float(), dim=-1)
except Exception:
    import clip  
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=("cuda" if torch.cuda.is_available() else "cpu"))
    CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.eval()
    def _clip_encode(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = clip_model.encode_image(img_batch.to(CLIP_DEVICE))
        return F.normalize(z.float(), dim=-1)

class ResizeLongEdge:
    def __init__(self, size=1024, interpolation=Image.BICUBIC):
        self.size = size; self.interpolation = interpolation
    def __call__(self, img):
        w, h = img.size
        if h > w:
            new_h = self.size; new_w = round(w * (new_h / h))
        else:
            new_w = self.size; new_h = round(h * (new_w / w))
        return img.resize((new_w, new_h), self.interpolation)

img_tf = transforms.Compose([
    ResizeLongEdge(1024),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageFilesWithMasks(Dataset):
    img_exts = {".jpg", ".jpeg", ".webp", ".bmp", ".png"}
    def __init__(self, root, tf):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(root)
        self.samples, self.mask_paths = [], []
        for p in root.rglob("*"):
            if (p.is_file()
                and not p.name.startswith(".")
                and "-checkpoint" not in p.name
                and p.suffix.lower() in self.img_exts
                and not p.name.endswith(MASK_SUFFIX)
                and ".ipynb_checkpoints" not in str(p)):
                m = p.with_name(p.stem + MASK_SUFFIX)
                if m.exists():
                    self.samples.append(str(p))
                    self.mask_paths.append(str(m))
                else:
                    print(f"Warning: skipping {p} (missing mask {m})")
        if not self.samples:
            raise RuntimeError(f"No images with masks found in {root}")
        self.tf = tf
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        with Image.open(self.samples[i]) as im:
            x = im.convert("RGB")
        return self.tf(x), self.samples[i], self.mask_paths[i]

class ImageFiles(Dataset):
    img_exts = {".jpg", ".jpeg", ".webp", ".bmp", ".png"}
    def __init__(self, root, tf):
        root = Path(root)
        if not root.is_dir(): raise FileNotFoundError(root)
        self.paths = []
        for p in root.rglob("*"):
            if (p.is_file()
                and not p.name.startswith(".")
                and "-checkpoint" not in p.name
                and p.suffix.lower() in self.img_exts
                and not p.name.endswith(MASK_SUFFIX)
                and ".ipynb_checkpoints" not in str(p)):
                self.paths.append(str(p))
        if not self.paths: raise RuntimeError(f"No images in {root}")
        self.tf = tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = im.convert("RGB")
        return self.tf(x), self.paths[i]

def collate_paths_and_masks(batch):
    xs, paths, mask_paths = zip(*batch)
    xs = torch.stack(xs, 0)
    return xs, list(paths), list(mask_paths)

def collate_with_paths_only(batch):
    xs, paths = zip(*batch)
    xs = torch.stack(xs, 0); return xs, list(paths)

gallery_ds = ImageFilesWithMasks(GALLERY_DIR, img_tf)
query_ds   = ImageFiles(QUERY_DIR, img_tf)

gallery_loader = DataLoader(gallery_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            collate_fn=collate_paths_and_masks)
query_loader   = DataLoader(query_ds, batch_size=1, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            collate_fn=collate_with_paths_only)

gallery_paths      = list(gallery_ds.samples)      
gallery_mask_paths = list(gallery_ds.mask_paths)   
query_paths        = list(query_ds.paths)          

@torch.no_grad()
def clip_embed_paths(paths: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = []
    buf = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
        buf.append(clip_preprocess(im).unsqueeze(0))  
        if len(buf) == batch_size:
            x = torch.cat(buf, 0) 
            v = _clip_encode(x)    
            vecs.append(v.cpu())
            buf = []
    if buf:
        x = torch.cat(buf, 0)
        v = _clip_encode(x)
        vecs.append(v.cpu())
    return torch.cat(vecs, 0).cpu().numpy().astype("float32")  

print("Computing CLIP embeddings…")
gallery_clip = clip_embed_paths(gallery_paths)  
query_clip   = clip_embed_paths(query_paths)    
print(f"Gallery: {len(gallery_paths)} | Query: {len(query_paths)} | Dim: {gallery_clip.shape[1]}")

def topk_cosine_numpy(q_vec: np.ndarray, matrix: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Both q_vec and matrix must be L2-normalized."""
    sims = (q_vec @ matrix.T).ravel()
    K = min(K, sims.size)
    idx  = np.argpartition(sims, -K)[-K:]
    idx  = idx[np.argsort(sims[idx])[::-1]]
    return sims[idx], idx

class NumpyIPIndex:
    """Lightweight IP (cosine on normalized vectors) index with FAISS-like .search()."""
    def __init__(self, xb: np.ndarray):
        self.xb = np.ascontiguousarray(xb, dtype="float32")
        self.ntotal = self.xb.shape[0]
    def search(self, q: np.ndarray, K: int):
        sim = q @ self.xb.T  
        K = min(K, self.ntotal)
        idx = np.argpartition(sim, -K, axis=1)[:, -K:]
        part = np.take_along_axis(sim, idx, axis=1)
        order = np.argsort(-part, axis=1)
        I = np.take_along_axis(idx, order, axis=1)
        D = np.take_along_axis(part, order, axis=1)
        return D.astype("float32"), I.astype("int64")

@torch.no_grad()
def _encode_image_tensor(model, x: torch.Tensor) -> torch.Tensor:
    out = model.image_encoder(x) 
    if torch.is_tensor(out):
        z = out
        if z.dim() == 3: z = z.unsqueeze(0)
        if z.dim() == 4: return z
    if isinstance(out, dict):
        for k in ("image_embed","image_embeddings","image_embeds","embeddings","features","feats"):
            if k in out:
                z = out[k]
                if isinstance(z, (list,tuple)):
                    cand = None
                    for t in reversed(z):
                        if torch.is_tensor(t) and t.dim()==4: cand=t; break
                    if cand is not None: return cand
                    if torch.is_tensor(z[-1]): z = z[-1]
                if torch.is_tensor(z):
                    if z.dim()==3: z=z.unsqueeze(0)
                    if z.dim()==4: return z
        for v in out.values():
            if torch.is_tensor(v) and v.dim()==4: return v
            if isinstance(v,(list,tuple)):
                for t in reversed(v):
                    if torch.is_tensor(t) and t.dim()==4: return t
    if isinstance(out,(list,tuple)):
        for t in reversed(out):
            if torch.is_tensor(t) and t.dim()==4: return t
    raise RuntimeError("image_encoder did not return a usable (B,C,Hf,Wf) feature map")

@torch.no_grad()
def feature_map_for_image(model, x: torch.Tensor) -> torch.Tensor:
    z = _encode_image_tensor(model, x) 
    return z[0]

def mask_to_feat(mask: np.ndarray, H: int, W: int) -> torch.Tensor:
    m = Image.fromarray(mask.astype(np.uint8)).resize((W, H), Image.NEAREST)
    m = np.array(m) > 0
    return torch.from_numpy(m.astype(np.float32)).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def masked_mean(feat: torch.Tensor, mask1: torch.Tensor) -> torch.Tensor:
    denom = mask1.sum().clamp_min(1.0)
    vec = (feat * mask1).view(feat.shape[0], -1).sum(dim=1) / denom
    return F.normalize(vec, dim=0)

def get_true_mask_idx(path: str) -> Optional[np.ndarray]:
    mask_path = Path(path).with_name(Path(path).stem + MASK_SUFFIX)
    if mask_path.exists():
        with Image.open(mask_path) as m:
            rgb = np.array(m.convert("RGB"))
        idx = rgb_mask_to_class_indices(rgb)
        return idx
    return None

def postprocess_masks_safe(low_res_masks, orig_hw):
    m = torch.as_tensor(low_res_masks, dtype=torch.float32, device=DEVICE)
    if m.ndim == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.ndim == 3:
        m = m.unsqueeze(0)
    elif m.ndim != 4:
        raise ValueError(f"Unexpected mask rank {m.ndim}")
    H, W = int(orig_hw[0]), int(orig_hw[1])
    return F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)

def preprocess_for_sam2_torch(image_np: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(image_np)
    x = img_tf(pil).unsqueeze(0).to(DEVICE)
    return x

def _collect_feature_tensors(obj, feats4d, feats3d):
    if torch.is_tensor(obj):
        if obj.dim() == 4: feats4d.append(obj)
        elif obj.dim() == 3: feats3d.append(obj)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj: _collect_feature_tensors(v, feats4d, feats3d); return
    if isinstance(obj, dict):
        for v in obj.values(): _collect_feature_tensors(v, feats4d, feats3d); return

def _safe_channel_reduce(x: torch.Tensor, c_target: int) -> torch.Tensor:
    B, C, H, W = x.shape
    if C == c_target: return x
    if C > c_target and C % c_target == 0:
        g = C // c_target
        return x.view(B, c_target, g, H, W).mean(dim=2)
    if C > c_target: return x[:, :c_target, :, :]
    pad = c_target - C
    return torch.cat([x, x.new_zeros(B, pad, H, W)], dim=1)

def _pick_resize_highres(feats4d: List[torch.Tensor], image_embeddings: torch.Tensor):
    device = image_embeddings.device
    B, C, Hf, Wf = image_embeddings.shape
    target_shapes = [(Hf * 4, Wf * 4), (Hf * 2, Wf * 2)]
    target_chans  = [C // 8, C // 4]
    def spatial_score(t: torch.Tensor, sH: int, sW: int) -> float:
        th, tw = t.shape[-2], t.shape[-1]
        rH, rW = th / max(sH, 1e-6), tw / max(sW, 1e-6)
        return abs(math.log2(max(rH, 1e-6))) + abs(math.log2(max(rW, 1e-6)))
    pool = list(feats4d)
    chosen = []
    for (tH, tW), tC in zip(target_shapes, target_chans):
        if pool:
            best_idx = min(range(len(pool)), key=lambda i: spatial_score(pool[i], tH, tW))
            x = pool.pop(best_idx)
        else:
            x = image_embeddings
        if x.dim() == 3: x = x.unsqueeze(0)
        if x.shape[0] != B:
            x = x[:1].repeat(B, 1, 1, 1)
        if (x.shape[-2], x.shape[-1]) != (tH, tW):
            x = F.interpolate(x.to(device), size=(tH, tW), mode="bilinear", align_corners=False)
        else:
            x = x.to(device)
        x = _safe_channel_reduce(x, tC)
        chosen.append(x)
    return chosen

def encode_image_with_grad(model, x: torch.Tensor):
    out = model.image_encoder(x)
    feats4d, feats3d = [], []
    _collect_feature_tensors(out, feats4d, feats3d)
    if not feats4d and feats3d:
        feats4d = [t.unsqueeze(0) for t in feats3d]
    if not feats4d and isinstance(out, dict):
        for k in ("image_embed","image_embeddings","image_embeds","embeddings","features","feats"):
            v = out.get(k, None)
            if torch.is_tensor(v):
                feats4d = [v.unsqueeze(0) if v.dim()==3 else v] if v.dim() in (3,4) else feats4d
                break
    if not feats4d:
        raise RuntimeError("encode_image_with_grad: couldn't find any 3D/4D tensors from image_encoder")
    feats4d = [t if t.dim()==4 else t.unsqueeze(0) for t in feats4d]
    feats4d_sorted = sorted(feats4d, key=lambda t: t.shape[-2]*t.shape[-1])
    image_embeddings = feats4d_sorted[0].to(x.device)
    high_res_features = _pick_resize_highres(feats4d, image_embeddings)
    return image_embeddings, high_res_features

for p in sam2_model.parameters(): p.requires_grad = True
for p in sam2_model.image_encoder.parameters(): p.requires_grad = True
for p in prompt_enc.parameters(): p.requires_grad = True
for p in mask_dec.parameters(): p.requires_grad = True

if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
else:
    from torch.cuda.amp import GradScaler as CudaGradScaler
    scaler = CudaGradScaler(enabled=torch.cuda.is_available())

def autocast_ctx():
    return torch.autocast(device_type="cuda", dtype=torch.float16) if torch.cuda.is_available() else nullcontext()

def snapshot_base_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def restore_base_state(model, base_state):
    model.load_state_dict(base_state, strict=True)

def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def ttft_make_optimizer(model, lr, wd):
    return optim.AdamW(trainable_params(model), lr=lr, weight_decay=wd)

BASE_STATE = snapshot_base_state(sam2_model)

def encode_points_to_prompts(image_np: np.ndarray, input_point: np.ndarray, input_labels: np.ndarray):
    """
    input_point: (N,1,2) or (N,2)
    input_labels: (N,1) or (N,), values in {0,1}
    """
    if input_point.ndim == 2:
        input_point = input_point.reshape(-1, 1, 2)
    if input_labels.ndim == 1:
        input_labels = input_labels.reshape(-1, 1)
    try:
        predictor.set_image(image_np)
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point, input_labels, box=None, mask_logits=None, normalize_coords=True
        )
        sparse_embeddings, dense_embeddings = prompt_enc(points=(unnorm_coords, labels), boxes=None, masks=None)
        return sparse_embeddings, dense_embeddings
    except Exception:
        H, W = image_np.shape[:2]
        coords = torch.as_tensor(input_point, dtype=torch.float32, device=DEVICE)
        coords[..., 0] = coords[..., 0] / float(W)
        coords[..., 1] = coords[..., 1] / float(H)
        labels_t = torch.as_tensor(input_labels.reshape(-1,1), dtype=torch.int64, device=DEVICE)
        sparse_embeddings, dense_embeddings = prompt_enc(points=(coords, labels_t), boxes=None, masks=None)
        return sparse_embeddings, dense_embeddings

@torch.no_grad()
def clip_targets_from_multicrops(img_path: str, n_crops: int = ITL_TARGET_PATCHES,
                                 min_side: int = 96, scale_range=(0.5, 1.0)) -> torch.Tensor:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        W, H = im.size

    rng = np.random.default_rng(0)
    crops = []
    for _ in range(n_crops):
        s = float(rng.uniform(*scale_range))
        cw, ch = max(min_side, int(W * s)), max(min_side, int(H * s))
        if cw >= W or ch >= H:
            x0, y0 = 0, 0
            cw, ch = W, H
        else:
            x0 = int(rng.integers(0, W - cw + 1))
            y0 = int(rng.integers(0, H - ch + 1))
        crops.append((x0, y0, x0 + cw, y0 + ch))
    crops.append((0, 0, W, H))

    tensors = []
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        for (x0, y0, x1, y1) in crops:
            crop = im.crop((x0, y0, x1, y1))
            tensors.append(clip_preprocess(crop).unsqueeze(0))
    x = torch.cat(tensors, 0)
    feats = _clip_encode(x)  
    return feats.cpu()

def predict_best_mask_from_points(image_np: np.ndarray,
                                  points_xy: np.ndarray,
                                  point_labels: np.ndarray,
                                  multimask_output: bool = True) -> Tuple[np.ndarray, float]:
    """
    Returns (mask_bool, score_float) using point prompts.
    Supports positives (1) and negatives (0).
    """
    H, W = image_np.shape[:2]
    try:
        predictor.set_image(image_np)
        masks, scores, _ = predictor.predict(
            point_coords=points_xy.astype(np.float32),
            point_labels=point_labels.astype(np.int32),
            box=None,
            multimask_output=multimask_output,
        )
        scores = np.asarray(scores).reshape(-1)
        j = int(np.argmax(scores))
        return (masks[j] > 0.5), float(scores[j])
    except Exception:
        input_point = points_xy.reshape(-1, 1, 2).astype(np.float32)  
        input_labels = point_labels.reshape(-1, 1).astype(np.int64)   
        sparse_embeddings, dense_embeddings = encode_points_to_prompts(image_np, input_point, input_labels)
        x_t = preprocess_for_sam2_torch(image_np)
        image_embeddings, high_res_features = encode_image_with_grad(sam2_model, x_t)
        low_res_masks, prd_scores, *_ = mask_dec(
            image_embeddings=image_embeddings,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=(input_point.shape[0] > 1),
            high_res_features=high_res_features,
        )
        up = postprocess_masks_safe(low_res_masks, (H, W))
        if up.ndim == 4: up = up[:, 0]
        masks = torch.sigmoid(up).detach().cpu().numpy()
        if isinstance(prd_scores, torch.Tensor) and prd_scores.numel():
            scores = prd_scores[:, 0].detach().cpu().numpy()
        else:
            scores = np.ones(masks.shape[0], dtype=np.float32)
        j = int(np.argmax(scores))
        return (masks[j] > 0.5), float(scores[j])

def load_points_file_for_image(image_path: str):
    """
    Reads POINTS_DIR/<image_stem>.txt and groups rows by instance_id.
    Format per line:
        <instance_id> <class_id> <x> <y> <label>
    Returns a list of (cls0, points_xy[N,2], labels[N]) packs.
    """
    txt = POINTS_DIR / (Path(image_path).stem + ".txt")
    if not txt.exists():
        return []

    rows = []
    with open(txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 5:
                continue
            gid  = int(float(parts[0]))
            cls0 = int(float(parts[1]))
            x    = float(parts[2]); y = float(parts[3])
            lab  = int(float(parts[4]))
            lab  = 1 if lab > 0 else 0  
            rows.append((gid, cls0, x, y, lab))

    if not rows:
        return []

    by_gid = defaultdict(list)
    cls_for_gid = {}
    for gid, cls0, x, y, lab in rows:
        by_gid[gid].append((x, y, lab))
        cls_for_gid.setdefault(gid, cls0)

    packs = []
    for gid in sorted(by_gid.keys()):
        triplets = by_gid[gid]
        pts = np.array([(x, y) for (x, y, _) in triplets], dtype=np.float32)
        labs = np.array([lab for (_, _, lab) in triplets], dtype=np.int32)
        cls0 = int(cls_for_gid[gid])
        packs.append((cls0, pts, labs))
    return packs

t_retrievals, t_ttft, t_amg, t_assign, t_totals = [], [], [], [], []  
perimage_gt, perimage_pred = [], []
ttft_skips = 0

ALL_BITS_LOGS = []  

@dataclass
class PredInst:
    img: str
    cls: int
    score: float
    mask: np.ndarray

@dataclass
class GTInst:
    img: str
    cls: int
    mask: np.ndarray

ALL_PREDS: List[PredInst] = []
ALL_GTS:   List[GTInst]   = []

K1 = min(K_PRESELECT, gallery_clip.shape[0])
N  = min(TOPN, K1)
assert N > 0, "Need at least 1 gallery item"

RETRIEVED_IMAGES_DIR = Path("./Retrieved_images")
RETRIEVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

for qi, q_path in enumerate(query_paths, start=1):
    restore_base_state(sam2_model, BASE_STATE)
    predictor = SAM2ImagePredictor(sam2_model)
    t_start_total = time.perf_counter()

    t0 = time.perf_counter()
    qv_clip = np.ascontiguousarray(query_clip[qi-1:qi], dtype="float32")  
    scores1, prefilter_ids = topk_cosine_numpy(qv_clip, gallery_clip, K1)  

    target_tensor = clip_targets_from_multicrops(q_path, n_crops=ITL_TARGET_PATCHES).float().cpu()
    itl = ITL(
        target=target_tensor,
        noise_std=ITL_NOISE_STD,
        subsampled_target_frac=ITL_SUBSAMPLED_TARGET_FRAC,
        max_target_size=ITL_MAX_TARGET_SIZE,
        mini_batch_size=ITL_MINI_BATCH,
    )

    sub_embs  = np.ascontiguousarray(gallery_clip[prefilter_ids], dtype="float32")
    sub_index = NumpyIPIndex(sub_embs)

    try:
        retriever = Retriever(index=sub_index, acquisition_function=itl,
                              alpha=ALPHA, also_query_opposite=False)
        K2 = K1
        values2, idxs2, _, _ = retriever.search(qv_clip, N=K2, K=K2)
        if isinstance(values2, np.ndarray): values2 = values2.ravel().tolist()
        if isinstance(idxs2,   np.ndarray): idxs2   = idxs2.ravel().tolist()
        pairs = [(prefilter_ids[int(i)], float(s)) for i, s in zip(idxs2, values2)]
    except (TypeError, AttributeError) as e1:
        print(f"Warning: activeft acquisition failed on subset ({e1}); using Stage-1 cosine ranks.")
        pairs = list(zip(prefilter_ids.tolist(), scores1.tolist()))

    best_by_path: Dict[str, Tuple[int, float]] = {}
    for i, s in pairs:
        path = gallery_paths[i]
        if (path not in best_by_path) or (s > best_by_path[path][1]):
            best_by_path[path] = (i, s)

    unique_sorted = sorted(best_by_path.values(), key=lambda t: t[1], reverse=True)
    idxs_unique = [i for (i, s) in unique_sorted[:N]]
    if len(idxs_unique) < N:
        print(f"Note: after de-dup only {len(idxs_unique)} unique neighbors available (requested {N}).")

    query_name = Path(q_path).stem
    query_retrieved_dir = RETRIEVED_IMAGES_DIR / query_name
    query_retrieved_dir.mkdir(parents=True, exist_ok=True)
    selected_image_paths = [gallery_paths[i] for i in idxs_unique]
    print(f"Query {qi} ({query_name}): Saving {len(selected_image_paths)} retrieved images to {query_retrieved_dir}")
    for i, src_path in enumerate(selected_image_paths, 1):
        src_path = Path(src_path)
        dest_path = query_retrieved_dir / f"retrieved_{i:03d}_{src_path.name}"
        shutil.copy2(src_path, dest_path)

    t_retrieval = time.perf_counter() - t0

    class_to_supports: Dict[int, List[Tuple[str, str]]] = {}
    for i in idxs_unique:
        img_p = gallery_paths[i]
        m_p   = gallery_mask_paths[i]
        with Image.open(m_p) as mm:
            m_rgb = np.array(mm.convert("RGB"))
        m_idx = rgb_mask_to_class_indices(m_rgb) 
        present = np.unique(m_idx)
        for cls_id in present:
            if cls_id == 0:  
                continue
            c = int(cls_id - 1)  
            class_to_supports.setdefault(c, []).append((img_p, m_p))

    ft_data_q = [{"image": p, "annotation": m}
                 for supps in class_to_supports.values() for (p, m) in supps]

    t0 = time.perf_counter()
    bpp_per_iter, bpb_per_iter = [], []

    if len(ft_data_q) > 0 and TTFT_UPDATES > 0:
        sam2_model.train()
        optimizer = ttft_make_optimizer(sam2_model, TTFT_LR, TTFT_WD)
        optimizer.zero_grad(set_to_none=True)

        base_trainable = None
        if TTFT_MU_PROX > 0:
            base_trainable = [p.detach().clone() for p in trainable_params(sam2_model)]

        updates_done = 0
        total_tries = 0
        while updates_done < TTFT_UPDATES:
            micro, tries = 0, 0
            pending_bpp_vals, pending_bpb_vals = [] , []
            while micro < TTFT_ACCUM_STEPS and tries < MAX_EMPTY_TRIES:
                with autocast_ctx():
                    ent = ft_data_q[np.random.randint(len(ft_data_q))]
                    with Image.open(ent["image"]) as src_im:
                        arr_raw = np.asarray(src_im)
                        channels_raw = 1 if arr_raw.ndim == 2 else arr_raw.shape[2]
                        bytes_per_channel = arr_raw.dtype.itemsize
                        bytes_per_pixel = float(bytes_per_channel * channels_raw)
                        image = np.array(src_im.convert("RGB"))
                    with Image.open(ent["annotation"]) as am:
                        am_rgb = np.array(am.convert("RGB"))
                    ann_map = rgb_mask_to_class_indices(am_rgb)     
                    labeled_mask, num_components = connected_components(ann_map > 0)
                    if num_components == 0:
                        tries += 1; total_tries += 1; continue

                    masks, points = [], []
                    for comp_id in range(1, num_components + 1):
                        mask = (labeled_mask == comp_id).astype(np.float32)
                        fg = np.argwhere(mask > 0)
                        if len(fg) == 0: continue
                        pt = fg[np.random.choice(len(fg), 1, replace=False)][0, [1, 0]]  
                        masks.append(mask)
                        points.append([pt])
                    if not masks:
                        tries += 1; total_tries += 1; continue

                    masks = np.array(masks)            
                    input_point = np.array(points)     
                    input_labels = np.ones((input_point.shape[0], 1), dtype=np.int64)

                    sparse_embeddings, dense_embeddings = encode_points_to_prompts(image, input_point, input_labels)
                    x = preprocess_for_sam2_torch(image)
                    image_embeddings, high_res_features = encode_image_with_grad(sam2_model, x)

                    low_res_masks, prd_scores, *_ = mask_dec(
                        image_embeddings=image_embeddings,
                        image_pe=dense_pe,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                        repeat_image=(input_point.shape[0] > 1),
                        high_res_features=high_res_features,
                    )

                    orig_hw = image.shape[:2]
                    prd_masks = postprocess_masks_safe(low_res_masks, orig_hw)  
                    prd_mask  = torch.sigmoid(prd_masks[:, 0])                  

                    gt_mask = torch.from_numpy(masks).float().to(DEVICE)
                    seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6)
                               - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-6)).mean()

                    inter = (gt_mask * (prd_mask > 0.5)).sum(dim=(1,2))
                    iou = inter / (gt_mask.sum(dim=(1,2)) + (prd_mask > 0.5).sum(dim=(1,2)) - inter + 1e-6)
                    if isinstance(prd_scores, torch.Tensor) and prd_scores.numel() > 0:
                        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                    else:
                        score_loss = 0.0

                    loss = seg_loss + 0.05 * score_loss

                    if TTFT_MU_PROX > 0:
                        prox = 0.0
                        for (p, p0) in zip(trainable_params(sam2_model), base_trainable):
                            prox = prox + (p - p0).pow(2).sum()
                        loss = loss + 0.5 * TTFT_MU_PROX * prox


                scaler.scale(loss / max(1, TTFT_ACCUM_STEPS)).backward()
                micro += 1

            if micro == 0:
                print(f"Warning: No valid samples for TTFT update {updates_done + 1} in query {qi}; skipping.")
                ttft_skips += 1
                break

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params(sam2_model), max_norm=TTFT_CLIP_NORM)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates_done += 1

            bpp_per_iter.append(float(np.mean(pending_bpp_vals)))
            bpb_per_iter.append(float(np.mean(pending_bpb_vals)))

        if total_tries > 0:
            print(f"Note: {total_tries} empty tries in TTFT for query {qi}.")
        sam2_model.eval()
    else:
        ttft_skips += 1
    t_ttft_q = time.perf_counter() - t0

    prototypes: Dict[int, torch.Tensor] = {}
    if not POINTS_HAVE_CLASS_ID:
        for c, supports in class_to_supports.items():
            if not supports: continue
            vecs = []
            supp_xs = []
            supp_bin_masks = []
            for img_path, mpath in supports:
                with Image.open(img_path) as im:
                    img_np = np.array(im.convert("RGB"))
                x = preprocess_for_sam2_torch(img_np)
                supp_xs.append(x)
                with Image.open(mpath) as mm:
                    m_rgb = np.array(mm.convert("RGB"))
                m_idx = rgb_mask_to_class_indices(m_rgb)
                supp_bin_masks.append((m_idx == (c+1)))
            supp_xs = torch.cat(supp_xs, dim=0)  
            feats = _encode_image_tensor(sam2_model, supp_xs)  
            Hf, Wf = feats.shape[-2:]
            for mask_bool, feat in zip(supp_bin_masks, feats):
                m = mask_to_feat(mask_bool.astype(np.float32), Hf, Wf)
                if m.sum().item() == 0: continue
                vecs.append(masked_mean(feat, m).detach().cpu())
            if vecs:
                proto = torch.stack(vecs, 0).median(0).values  
                prototypes[c] = F.normalize(proto.to(DEVICE), dim=0)

    t0 = time.perf_counter()
    with Image.open(q_path) as qim:
        q_img = np.array(qim.convert("RGB"))
    x_q = preprocess_for_sam2_torch(q_img)  

    packs = load_points_file_for_image(q_path)
    if not packs:
        print(f"Warning: no points file found for {q_path}; skipping image.")
        t_amg_q = 0.0
        t_assign_q = time.perf_counter() - t0
        t_total = time.perf_counter() - t_start_total
        t_retrievals.append(t_retrieval); t_ttft.append(t_ttft_q)
        t_amg.append(t_amg_q); t_assign.append(t_assign_q); t_totals.append(t_total)
        continue

    H, W = q_img.shape[:2]
    score_map = np.full((H, W), -np.inf, dtype=np.float32)
    class_map = np.zeros((H, W), dtype=np.uint8)
    preds_for_image: List[PredInst] = []

    for cls0, pts, labs in packs:
        if pts.size == 0:
            continue
        if POINTS_COORDS_NORMALIZED:
            pts = pts.copy()
            pts[:, 0] *= W
            pts[:, 1] *= H

        num_pos = int(labs.sum())
        multimask = True if (pts.shape[0] == 1 and num_pos == 1) else False

        mask_bin, score = predict_best_mask_from_points(
            q_img, pts.astype(np.float32), labs.astype(np.int32), multimask_output=multimask
        )

        if POINTS_HAVE_CLASS_ID:
            cls_id = int(cls0) + POINTS_CLASS_OFFSET
            cls_id = max(1, min(NUM_CLASSES, cls_id)) 
        else:
            if len(prototypes) > 0:
                q_feat = feature_map_for_image(sam2_model, x_q)
                m_feat = mask_to_feat(mask_bin.astype(np.float32), q_feat.shape[-2], q_feat.shape[-1])
                vec_q = masked_mean(q_feat, m_feat)
                best_c, best_sim = -1, -1.0
                for c, proto in prototypes.items():
                    sim = float(torch.dot(vec_q, proto))
                    if sim > best_sim:
                        best_sim, best_c = sim, c
                cls_id = best_c + 1 if (best_c >= 0 and best_sim >= SIM_THRESHOLD) else 1
            else:
                cls_id = 1 

        preds_for_image.append(PredInst(img=q_path, cls=cls_id, mask=mask_bin, score=score))

        if RESOLVE_TIES_HIGHER:
            upd = mask_bin & (score > score_map)
        else:
            upd = mask_bin & (score_map == -np.inf)
        score_map[upd] = score
        class_map[upd] = cls_id

    t_amg_q = 0.0 
    t_assign_q = time.perf_counter() - t0 

    q_stem = Path(q_path).stem
    raw_mask_path  = MASK_OUT_DIR / f"{q_stem}_labels.png"
    viz_mask_path  = MASK_OUT_DIR / f"{q_stem}_viz.png"
    class_map_u8 = class_map.astype(np.uint8)
    Image.fromarray(class_map_u8).save(raw_mask_path)
    save_color_mask_idx(class_map_u8, viz_mask_path)

    true_mask = get_true_mask_idx(q_path)

    ALL_PREDS.extend(preds_for_image)
    if true_mask is not None:
        perimage_gt.append(true_mask)
        perimage_pred.append(class_map)
        for cls_id in range(1, NUM_CLASSES+1):
            lab, ncomp = connected_components(true_mask == cls_id)
            for k in range(1, ncomp + 1):
                gt_m = (lab == k)
                if gt_m.sum() == 0: continue
                ALL_GTS.append(GTInst(img=q_path, cls=cls_id, mask=gt_m))

    t_total = time.perf_counter() - t_start_total
    t_retrievals.append(t_retrieval); t_ttft.append(t_ttft_q)
    t_amg.append(t_amg_q); t_assign.append(t_assign_q); t_totals.append(t_total)

    print(f"[{qi:03d}] {q_stem} -> {raw_mask_path.name}, {viz_mask_path.name} | "
          f"ret={t_retrieval:.3f}s, ttft={t_ttft_q:.3f}s, pointseg={(t_amg_q + t_assign_q):.3f}s, e2e={t_total:.3f}s")

def per_class_intersection_union(gt: np.ndarray, pr: np.ndarray, class_ids: List[int]):
    inter, uni, supp = {}, {}, {}
    for c in class_ids:
        g = (gt == c)
        p = (pr == c)
        i = int(np.logical_and(g, p).sum())
        u = int(g.sum() + p.sum() - i)
        inter[c] = i
        uni[c] = u
        supp[c] = int(g.sum())
    return inter, uni, supp

def aggregate_over_dataset(gts: List[np.ndarray], prs: List[np.ndarray], class_ids: List[int]):
    tot_inter = defaultdict(int)
    tot_union = defaultdict(int)
    tot_support = defaultdict(int)
    for gt, pr in zip(gts, prs):
        inter, uni, supp = per_class_intersection_union(gt, pr, class_ids)
        for c in class_ids:
            tot_inter[c] += inter[c]
            tot_union[c] += uni[c]
            tot_support[c] += supp[c]
    return tot_inter, tot_union, tot_support

def compute_dataset_mIoU_and_weighted(gts: List[np.ndarray], prs: List[np.ndarray],
                                      class_ids: List[int]):
    tot_inter, tot_union, tot_support = aggregate_over_dataset(gts, prs, class_ids)
    iou_per_class = {}
    for c in class_ids:
        if tot_union[c] > 0:
            iou_per_class[c] = tot_inter[c] / float(tot_union[c])
        else:
            iou_per_class[c] = None
    valid = [v for v in iou_per_class.values() if v is not None]
    macro = float(np.mean(valid)) if valid else 0.0
    total_gt = sum(tot_support[c] for c in class_ids)
    if total_gt > 0:
        weighted = float(sum(tot_support[c] * (iou_per_class[c] if iou_per_class[c] is not None else 0.0)
                             for c in class_ids) / total_gt)
    else:
        weighted = 0.0
    return macro, weighted, iou_per_class

def compute_dataset_dice(gts: List[np.ndarray], prs: List[np.ndarray], class_ids: List[int]):
    inter = defaultdict(int); gt_sum = defaultdict(int); pr_sum = defaultdict(int)
    for gt, pr in zip(gts, prs):
        for c in class_ids:
            g = (gt == c); p = (pr == c)
            inter[c] += int(np.logical_and(g, p).sum())
            gt_sum[c] += int(g.sum())
            pr_sum[c] += int(p.sum())
    dice_per = {}
    for c in class_ids:
        denom = gt_sum[c] + pr_sum[c]
        dice_per[c] = (2.0 * inter[c] / denom) if denom > 0 else None
    valid = [v for v in dice_per.values() if v is not None]
    mdice = float(np.mean(valid)) if valid else 0.0
    return mdice, dice_per

def compute_dataset_prf1(gts: List[np.ndarray], prs: List[np.ndarray], class_ids: List[int]):
    TP = defaultdict(int); FP = defaultdict(int); FN = defaultdict(int)
    for gt, pr in zip(gts, prs):
        for c in class_ids:
            g = (gt == c)
            p = (pr == c)
            TP[c] += int(np.logical_and(g, p).sum())
            FP[c] += int(np.logical_and(~g, p).sum())
            FN[c] += int(np.logical_and(g, ~p).sum())
    P, R, F1 = {}, {}, {}
    for c in class_ids:
        p = TP[c] / float(TP[c] + FP[c]) if (TP[c] + FP[c]) > 0 else None
        r = TP[c] / float(TP[c] + FN[c]) if (TP[c] + FN[c]) > 0 else None
        f = (2*p*r)/(p+r) if (p is not None and r is not None and (p+r) > 0) else None
        P[c], R[c], F1[c] = p, r, f
    def _macro(d):
        vals = [v for v in d.values() if v is not None]
        return float(np.mean(vals)) if vals else 0.0
    return P, R, F1, _macro(P), _macro(R), _macro(F1)

def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    er = binary_erosion(mask, iterations=1, border_value=0)
    return np.logical_xor(mask.astype(bool), er.astype(bool))

def _bf1_counts(gt_b: np.ndarray, pr_b: np.ndarray, tol: int) -> Tuple[int, int, int]:
    if pr_b.any():
        dist_to_gt = distance_transform_edt(~gt_b)
        tp_prec = int((dist_to_gt[pr_b] <= tol).sum())
        fp = int(pr_b.sum()) - tp_prec
    else:
        tp_prec, fp = 0, 0
    if gt_b.any():
        dist_to_pr = distance_transform_edt(~pr_b)
        tp_rec = int((dist_to_pr[gt_b] <= tol).sum())
        fn = int(gt_b.sum()) - tp_rec
    else:
        tp_rec, fn = 0, 0
    tp = int(0.5 * (tp_prec + tp_rec))
    return tp, fp, fn

def compute_boundary_prf1(gts: List[np.ndarray], prs: List[np.ndarray], class_ids: List[int], tolerance: int = 2):
    TP = defaultdict(int); FP = defaultdict(int); FN = defaultdict(int)
    for gt, pr in zip(gts, prs):
        for c in class_ids:
            g = (gt == c); p = (pr == c)
            gb = _binary_boundary(g); pb = _binary_boundary(p)
            tp, fp, fn = _bf1_counts(gb, pb, tol=tolerance)
            TP[c] += tp; FP[c] += fp; FN[c] += fn
    P, R, F1 = {}, {}, {}
    for c in class_ids:
        p = TP[c] / float(TP[c] + FP[c]) if (TP[c] + FP[c]) > 0 else None
        r = TP[c] / float(TP[c] + FN[c]) if (TP[c] + FN[c]) > 0 else None
        f = (2*p*r)/(p+r) if (p is not None and r is not None and (p+r) > 0) else None
        P[c], R[c], F1[c] = p, r, f
    def _macro(d):
        vals = [v for v in d.values() if v is not None]
        return float(np.mean(vals)) if vals else 0.0
    return P, R, F1, _macro(P), _macro(R), _macro(F1)

@dataclass
class PredInst:
    img: str
    cls: int
    score: float
    mask: np.ndarray

@dataclass
class GTInst:
    img: str
    cls: int
    mask: np.ndarray

def _rle_from_binary_mask(m: np.ndarray) -> dict:
    m = np.asfortranarray(m.astype(np.uint8))
    rle = maskUtils.encode(m)
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle

def _bbox_from_mask(m: np.ndarray) -> List[float]:
    ys, xs = np.where(m > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]

def build_coco_dict(all_gts: List[GTInst], all_preds: List[PredInst]) -> Tuple[dict, list]:
    img_ids = {}
    images = []
    next_id = 1
    sizes_cache = {}
    for inst in all_gts:
        if inst.img not in img_ids:
            with Image.open(inst.img) as im:
                w, h = im.size
            img_ids[inst.img] = next_id; next_id += 1
            images.append({"id": img_ids[inst.img], "file_name": Path(inst.img).name, "width": w, "height": h})
            sizes_cache[inst.img] = (w, h)
    for inst in all_preds:
        if inst.img not in img_ids:
            with Image.open(inst.img) as im:
                w, h = im.size
            img_ids[inst.img] = next_id; next_id += 1
            images.append({"id": img_ids[inst.img], "file_name": Path(inst.img).name, "width": w, "height": h})
            sizes_cache[inst.img] = (w, h)

    categories = [{"id": i, "name": CLASS_NAMES[i-1]} for i in range(1, NUM_CLASSES+1)]

    annotations = []
    ann_id = 1
    for g in all_gts:
        rle = _rle_from_binary_mask(g.mask)
        bbox = _bbox_from_mask(g.mask)
        area = float(maskUtils.area(rle))
        annotations.append({
            "id": ann_id,
            "image_id": img_ids[g.img],
            "category_id": int(g.cls),
            "segmentation": rle,
            "iscrowd": 0,
            "bbox": bbox,
            "area": area,
        })
        ann_id += 1

    coco_gt = {"images": images, "annotations": annotations, "categories": categories}

    coco_preds = []
    for p in all_preds:
        rle = _rle_from_binary_mask(p.mask)
        bbox = _bbox_from_mask(p.mask)
        area = float(maskUtils.area(rle))
        coco_preds.append({
            "image_id": img_ids[p.img],
            "category_id": int(p.cls),
            "segmentation": rle,
            "score": float(p.score),
            "bbox": bbox,
            "area": area,
        })
    return coco_gt, coco_preds

def _per_class_ap_from_cocoeval(ev: COCOeval) -> Tuple[Dict[int,float], Dict[int,float], Dict[int,float]]:
    precisions = ev.eval['precision'] 
    if precisions is None:
        return {}, {}, {}
    T, R, K, A, M = precisions.shape
    iou_thrs = np.linspace(0.5, 0.95, 10)
    i50 = int(np.where(np.isclose(iou_thrs, 0.5))[0][0])
    i75 = int(np.where(np.isclose(iou_thrs, 0.75))[0][0])

    ap_per_class = {}
    ap50_per_class = {}
    ap75_per_class = {}
    for k in range(K):
        pk = precisions[:, :, k, :, :]
        pk = pk[pk > -1]
        ap_per_class[k+1] = float(pk.mean()) if pk.size else 0.0

        p50 = ev.eval['precision'][i50, :, k, :, :]
        p50 = p50[p50 > -1]
        ap50_per_class[k+1] = float(p50.mean()) if p50.size else 0.0

        p75 = ev.eval['precision'][i75, :, k, :, :]
        p75 = p75[p75 > -1]
        ap75_per_class[k+1] = float(p75.mean()) if p75.size else 0.0
    return ap_per_class, ap50_per_class, ap75_per_class

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union if union > 0 else 1)

def ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    recall_thrs = np.linspace(0, 1, 101)
    prec_at_rec = []
    for t in recall_thrs:
        inds = np.where(mrec >= t)[0]
        prec_at_rec.append(mpre[inds[0]] if inds.size else 0.0)
    return float(np.mean(prec_at_rec))

def eval_ap_for_threshold(iou_thr: float, preds: List[PredInst], gts: List[GTInst], classes: List[int]):
    ap_per_cls: Dict[int, float] = {}
    gts_by_img_cls: Dict[Tuple[str, int], List[GTInst]] = defaultdict(list)
    for g in gts: gts_by_img_cls[(g.img, g.cls)].append(g)
    for cls in classes:
        P = [p for p in preds if p.cls == cls]
        G = [g for g in gts if g.cls == cls]
        if len(G) == 0: 
            ap_per_cls[cls] = 0.0
            continue
        P.sort(key=lambda x: x.score, reverse=True)
        tp = np.zeros(len(P), dtype=np.float32)
        fp = np.zeros(len(P), dtype=np.float32)
        matched = {id(g): False for g in G}
        for i, p in enumerate(P):
            cands = gts_by_img_cls.get((p.img, cls), [])
            best_iou, best_gid = 0.0, None
            for g in cands:
                if matched[id(g)]: continue
                iou = mask_iou(p.mask, g.mask)
                if iou > best_iou:
                    best_iou, best_gid = iou, id(g)
            if best_iou >= iou_thr and best_gid is not None:
                tp[i] = 1.0; matched[best_gid] = True
            else:
                fp[i] = 1.0
        tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(len(G), 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap_per_cls[cls] = ap_from_pr(recalls, precisions)
    mAP = float(np.mean(list(ap_per_cls.values()))) if ap_per_cls else 0.0
    return ap_per_cls, mAP

print("\n=== Evaluation===")

macro_miou = weighted_iou = mdice = None
iou_per_class = dice_per_class = None

if len(perimage_gt) == 0:
    print("No ground-truth masks found for queries. Add query masks for eval.")
else:
    CLASS_IDS = list(range(1, NUM_CLASSES+1))  # 1..14

    macro_miou, weighted_iou, iou_per_class = compute_dataset_mIoU_and_weighted(
        [gt.astype(np.int32) for gt in perimage_gt],
        [pr.astype(np.int32) for pr in perimage_pred],
        CLASS_IDS
    )
    mdice, dice_per_class = compute_dataset_dice(
        [gt.astype(np.int32) for gt in perimage_gt],
        [pr.astype(np.int32) for pr in perimage_pred],
        CLASS_IDS
    )
    print(f"Images evaluated: {len(perimage_gt)}")
    print(f"mIoU (macro over 14 classes)   : {macro_miou:.4f}")
    print(f"Freq-weighted IoU (14 classes) : {weighted_iou:.4f}")
    print(f"mDice (macro over 14 classes)  : {mdice:.4f}")
    for c in CLASS_IDS:
        iou_c = iou_per_class[c]
        dice_c = dice_per_class[c]
        cname = CLASS_NAMES[c-1]
        iou_str  = f"{iou_c:.4f}"  if iou_c  is not None else "n/a"
        dice_str = f"{dice_c:.4f}" if dice_c is not None else "n/a"
        print(f"  class {c:>2} ({cname}): IoU={iou_str} | Dice={dice_str}")

    P_std, R_std, F1_std, P_std_macro, R_std_macro, F1_std_macro = compute_dataset_prf1(
        [gt.astype(np.int32) for gt in perimage_gt],
        [pr.astype(np.int32) for pr in perimage_pred],
        CLASS_IDS
    )

    BF1_TOL = 2
    P_b, R_b, F1_b, P_b_macro, R_b_macro, F1_b_macro = compute_boundary_prf1(
        [gt.astype(np.int32) for gt in perimage_gt],
        [pr.astype(np.int32) for pr in perimage_pred],
        CLASS_IDS,
        tolerance=BF1_TOL
    )

    ap_metrics = None
    if len(ALL_GTS) == 0:
        print("\n=== COCO-style Instance AP ===\nNo GT instances available for AP; skipping.")
    else:
        if _HAS_COCO:
            coco_gt_dict, coco_pred_list = build_coco_dict(ALL_GTS, ALL_PREDS)
            coco_gt_api = COCO()
            coco_gt_api.dataset = coco_gt_dict
            coco_gt_api.dataset.setdefault("info", {"description": "temp"})
            coco_gt_api.dataset.setdefault("licenses", [])
            coco_gt_api.createIndex()
            coco_dt_api = coco_gt_api.loadRes(coco_pred_list)
            coco_eval = COCOeval(coco_gt_api, coco_dt_api, iouType='segm')
            coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()

            per_class_AP_all, per_class_AP50, per_class_AP75 = _per_class_ap_from_cocoeval(coco_eval)
            overall_mAP_50_95 = float(coco_eval.stats[0])
            overall_AP50 = float(coco_eval.stats[1])
            overall_AP75 = float(coco_eval.stats[2])

            ap_metrics = {
                "AP50": round(overall_AP50, 4),
                "AP75": round(overall_AP75, 4),
                "mAP_50_95": round(overall_mAP_50_95, 4),
                "per_class_AP50": {int(k): round(float(v), 4) for k, v in per_class_AP50.items()},
                "per_class_AP75": {int(k): round(float(v), 4) for k, v in per_class_AP75.items()},
                "per_class_mAP_50_95": {int(k): round(float(v), 4) for k, v in per_class_AP_all.items()},
            }
        else:
            print("[warn] Using internal AP fallback (not COCOeval).")
            iou_thrs = np.arange(0.50, 0.95 + 1e-9, 0.05)
            ap50_cls, mAP50 = eval_ap_for_threshold(0.50, ALL_PREDS, ALL_GTS, CLASS_IDS)
            ap75_cls, mAP75 = eval_ap_for_threshold(0.75, ALL_PREDS, ALL_GTS, CLASS_IDS)
            ap_by_thr = []
            for t in iou_thrs:
                _, mAP_t = eval_ap_for_threshold(float(t), ALL_PREDS, ALL_GTS, CLASS_IDS)
                ap_by_thr.append(mAP_t)
            mAP_50_95 = float(np.mean(ap_by_thr)) if ap_by_thr else 0.0
            ap_metrics = {
                "AP50": round(float(mAP50), 4),
                "AP75": round(float(mAP75), 4),
                "mAP_50_95": round(float(mAP_50_95), 4),
                "per_class_AP50": {int(k): round(float(v), 4) for k, v in ap50_cls.items()},
                "per_class_AP75": {int(k): round(float(v), 4) for k, v in ap75_cls.items()},
                "per_class_mAP_50_95": {}, 
            }

METRICS_DIR = OUT_DIR
METRICS_PATH = METRICS_DIR / "metrics_summary.json"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

_mean = lambda v: float(np.mean(v)) if len(v) else 0.0

summary = {
    "images_evaluated": int(len(perimage_gt)),
    "timing_sec_mean": {
        "retrieval": float(_mean(t_retrievals)),
        "ttft_per_query": float(_mean(t_ttft)),
        "amg_proposals": float(_mean(t_amg)),
        "assignment": float(_mean(t_assign)),
        "end_to_end": float(_mean(t_totals)),
    }
}

if macro_miou is not None:
    summary.update({
        "miou_macro_14": round(float(macro_miou), 4),
        "freq_weighted_iou_14": round(float(weighted_iou), 4),
        "mdice_macro_14": round(float(mdice), 4),
        "per_class_region": [
            {
                "id": int(c),
                "name": CLASS_NAMES[c - 1],
                "iou": None if iou_per_class[c] is None else round(float(iou_per_class[c]), 4),
                "dice": None if dice_per_class[c] is None else round(float(dice_per_class[c]), 4),
                "precision": None if P_std.get(c, None) is None else round(float(P_std[c]), 4),
                "recall": None if R_std.get(c, None) is None else round(float(R_std[c]), 4),
                "f1": None if F1_std.get(c, None) is None else round(float(F1_std[c]), 4),
            }
            for c in range(1, NUM_CLASSES + 1)
        ],
        "region_macro": {
            "precision": round(float(P_std_macro), 4),
            "recall": round(float(R_std_macro), 4),
            "f1": round(float(F1_std_macro), 4),
        },
    })

if len(perimage_gt) > 0:
    summary.update({
        "boundary_f1": {
            "tolerance_px": int(BF1_TOL),
            "per_class": [
                {
                    "id": int(c),
                    "name": CLASS_NAMES[c - 1],
                    "precision": None if P_b.get(c, None) is None else round(float(P_b[c]), 4),
                    "recall": None if R_b.get(c, None) is None else round(float(R_b[c]), 4),
                    "f1": None if F1_b.get(c, None) is None else round(float(F1_b[c]), 4),
                }
                for c in range(1, NUM_CLASSES + 1)
            ],
            "macro": {
                "precision": round(float(P_b_macro), 4),
                "recall": round(float(R_b_macro), 4),
                "f1": round(float(F1_b_macro), 4),
            },
        }
    })

if ap_metrics is not None:
    summary["instance_AP"] = ap_metrics

summary = json.loads(json.dumps(summary, default=_to_py))
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"[metrics] wrote JSON to {METRICS_PATH}")

print("\n=== Timing (mean per query) ===")
print(f"TTFT (per-query) : {_mean(t_ttft):.4f} s")
if ttft_skips > 0:
    print(f"Note: TTFT skipped for {ttft_skips} queries due to no valid samples.")
print("Done")
