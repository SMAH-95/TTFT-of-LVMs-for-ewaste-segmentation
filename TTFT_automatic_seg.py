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
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from activeft.sift import Retriever
from activeft.acquisition_functions.itl import ITL
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

#config
GALLERY_DIR = "./Dataset/candidate_images"
QUERY_DIR   = "./Dataset/query_images"
CKPT_PATH   = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_VERSION = "sam2.1"   # "sam2" or "sam2.1"
SAM2_VARIANT = "large"     # "tiny" | "small" | "base" | "large"
OUT_DIR = Path("./Results/automatic_seg")
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
AMG_POINTS_PER_SIDE = 32
AMG_PRED_IOU_THR    = 0.5
AMG_STAB_THR        = 0.9
AMG_MIN_REGION      = 0
AMG_CROP_LAYERS     = 0
SIM_THRESHOLD       = 0.50
RESOLVE_TIES_HIGHER = True
MIN_PROPOSAL_PIXELS = 16  
TTFT_UPDATES      = 15
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
ITL_TARGET_PATCHES = 64  # number of CLIP multi-crop targets

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

def encode_points_to_prompts(image_np: np.ndarray, input_point: np.ndarray):
    try:
        predictor.set_image(image_np)
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point, np.ones((input_point.shape[0], 1)), box=None, mask_logits=None, normalize_coords=True
        )
        sparse_embeddings, dense_embeddings = prompt_enc(points=(unnorm_coords, labels), boxes=None, masks=None)
        return sparse_embeddings, dense_embeddings
    except Exception:
        H, W = image_np.shape[:2]
        coords = torch.as_tensor(input_point, dtype=torch.float32, device=DEVICE)
        coords[..., 0] = coords[..., 0] / float(W)
        coords[..., 1] = coords[..., 1] / float(H)
        labels = torch.ones((coords.shape[0], 1), dtype=torch.int64, device=DEVICE)
        sparse_embeddings, dense_embeddings = prompt_enc(points=(coords, labels), boxes=None, masks=None)
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

t_retrievals, t_ttft, t_amg, t_assign, t_totals = [], [], [], [], []
perimage_gt, perimage_pred = [], []
ttft_skips = 0

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

try:
    amg = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=AMG_POINTS_PER_SIDE,
        pred_iou_thresh=AMG_PRED_IOU_THR,
        stability_score_thresh=AMG_STAB_THR,
        min_mask_region_area=AMG_MIN_REGION,
        crop_n_layers=AMG_CROP_LAYERS,
        multimask_output=True,
    )
except TypeError:
    amg = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=AMG_POINTS_PER_SIDE,
        pred_iou_thresh=AMG_PRED_IOU_THR,
        stability_score_thresh=AMG_STAB_THR,
        min_mask_region_area=AMG_MIN_REGION,
        crop_n_layers=AMG_CROP_LAYERS,
    )

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

                    sparse_embeddings, dense_embeddings = encode_points_to_prompts(image, input_point)
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

        if total_tries > 0:
            print(f"Note: {total_tries} empty tries in TTFT for query {qi}.")
        sam2_model.eval()
    else:
        ttft_skips += 1
    t_ttft_q = time.perf_counter() - t0

    prototypes: Dict[int, torch.Tensor] = {}
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
    proposals = amg.generate(q_img)
    t_amg_q = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds_for_image: List[PredInst] = []
    if len(prototypes) == 0:
        class_map = np.zeros(q_img.shape[:2], dtype=np.uint8)
    else:
        q_feat = feature_map_for_image(sam2_model, x_q)
        H, W = q_img.shape[:2]
        score_map = np.full((H, W), -np.inf, dtype=np.float32)
        class_map = np.zeros((H, W), dtype=np.uint8)

        for prop in proposals:
            seg = prop.get("segmentation")
            if isinstance(seg, torch.Tensor): seg = seg.detach().cpu().numpy()
            if seg is None: continue
            seg = (seg > 0)
            if seg.sum() < MIN_PROPOSAL_PIXELS: continue

            m_feat = mask_to_feat(seg.astype(np.float32), q_feat.shape[-2], q_feat.shape[-1])
            vec_q = masked_mean(q_feat, m_feat)
            best_c, best_sim = -1, -1.0
            for c, proto in prototypes.items():
                sim = float(torch.dot(vec_q, proto))
                if sim > best_sim:
                    best_sim, best_c = sim, c
            if best_c < 0 or best_sim < SIM_THRESHOLD: continue

            preds_for_image.append(PredInst(img=q_path, cls=best_c + 1, score=float(best_sim), mask=seg))

            if RESOLVE_TIES_HIGHER:
                upd = seg & (best_sim > score_map)
            else:
                upd = seg & (score_map == -np.inf)
            score_map[upd] = best_sim
            class_map[upd] = best_c + 1

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
          f"ttft={t_ttft_q:.3f}s")

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

def _binary_boundary(mask_bool: np.ndarray) -> np.ndarray:
    m = mask_bool.astype(bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    er = binary_erosion(m)
    return np.logical_and(m, np.logical_not(er))

def _bf_counts(pred_bin: np.ndarray, gt_bin: np.ndarray, tol_px: int = 2) -> Tuple[int, int, int, int]:
    pb = _binary_boundary(pred_bin)
    gb = _binary_boundary(gt_bin)

    if tol_px > 0:
        se = np.ones((2 * tol_px + 1, 2 * tol_px + 1), dtype=bool)
        gb_dil = binary_dilation(gb, structure=se)
        pb_dil = binary_dilation(pb, structure=se)
    else:
        gb_dil, pb_dil = gb, pb

    matched_pred = int(np.logical_and(pb, gb_dil).sum())
    matched_gt   = int(np.logical_and(gb, pb_dil).sum())
    total_pred   = int(pb.sum())
    total_gt     = int(gb.sum())
    return matched_pred, total_pred, matched_gt, total_gt

def compute_dataset_boundary_f1(
    gts: List[np.ndarray],
    prs: List[np.ndarray],
    class_ids: List[int],
    tol_px: int = 2,
) -> Tuple[float, Dict[int, Dict[str, Optional[float]]], Dict[str, float]]:
    acc = {int(c): {"mp": 0, "tp": 0, "mg": 0, "tg": 0} for c in class_ids}
    pool = {"mp": 0, "tp": 0, "mg": 0, "tg": 0}

    for gt, pr in zip(gts, prs):
        for c in class_ids:
            g_bin = (gt == c)
            p_bin = (pr == c)
            mp, tp, mg, tg = _bf_counts(p_bin, g_bin, tol_px=tol_px)
            a = acc[c]
            a["mp"] += mp; a["tp"] += tp
            a["mg"] += mg; a["tg"] += tg
            pool["mp"] += mp; pool["tp"] += tp
            pool["mg"] += mg; pool["tg"] += tg

    def _prf(d):
        P = (d["mp"] / d["tp"]) if d["tp"] > 0 else None
        R = (d["mg"] / d["tg"]) if d["tg"] > 0 else None
        if (P is None) or (R is None) or (P + R == 0):
            F1 = None
        else:
            F1 = 2 * P * R / (P + R)
        return P, R, F1

    per_class = {}
    for c in class_ids:
        P, R, F1 = _prf(acc[c])
        per_class[c] = {
            "P": None if P is None else float(P),
            "R": None if R is None else float(R),
            "F1": None if F1 is None else float(F1),
            "tot_pred_b": int(acc[c]["tp"]),
            "tot_gt_b": int(acc[c]["tg"]),
        }

    P_pool, R_pool, F1_pool = _prf(pool)
    overall = {
        "P": 0.0 if P_pool is None else float(P_pool),
        "R": 0.0 if R_pool is None else float(R_pool),
        "F1": 0.0 if F1_pool is None else float(F1_pool),
        "tot_pred_b": int(pool["tp"]),
        "tot_gt_b": int(pool["tg"]),
    }
    return overall["F1"], per_class, overall

def _to_coco_categories(names: List[str]) -> List[dict]:
    return [{"id": i+1, "name": names[i], "supercategory": "object"} for i in range(len(names))]

def _bbox_from_binary_mask(mask_hw: np.ndarray) -> List[float]:
    rle = maskUtils.encode(np.asfortranarray(mask_hw.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    x, y, w, h = maskUtils.toBbox(rle).tolist()
    return [float(x), float(y), float(w), float(h)]

def _rle_from_binary_mask(mask_hw: np.ndarray) -> dict:
    rle = maskUtils.encode(np.asfortranarray(mask_hw.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle

def build_coco_gt_and_dt(
    preds: List[dataclass],
    gts: List[dataclass],
    class_names: List[str],
    image_paths: List[str],
) -> Tuple[COCO, List[dict], Dict[str, int], Dict[int, str]]:
    """
    Returns:
      coco_gt: COCO object for GT
      coco_dt_list: list[annotation-like dict] for detections
      img_id_by_path: mapping from path -> image_id (int)
      cat_id_to_name: mapping category_id -> name
    """
    img_id_by_path = {p: i+1 for i, p in enumerate(image_paths)}
    images = []
    for p, img_id in img_id_by_path.items():
        with Image.open(p) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": os.path.basename(p), "width": int(w), "height": int(h)})

    categories = _to_coco_categories(class_names)
    cat_id_to_name = {c["id"]: c["name"] for c in categories}

    ann_id = 1
    annotations = []
    for g in gts:
        img_id = img_id_by_path[g.img]
        mask_bool = g.mask.astype(np.uint8)
        rle = _rle_from_binary_mask(mask_bool)
        bbox = maskUtils.toBbox(rle).tolist()
        area = float(maskUtils.area(rle))
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": int(g.cls),  
            "segmentation": rle,
            "area": area,
            "bbox": [float(b) for b in bbox],
            "iscrowd": 0
        })
        ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {
            "description": "E-waste instance segmentation GT",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Aseni",
            "date_created": datetime.now().strftime("%Y-%m-%d")
        },
        "licenses": [],
        "images": images,
        "categories": categories,
        "annotations": annotations
    }
    coco_gt.createIndex()

    coco_dt_list = []
    for p in preds:
        img_id = img_id_by_path[p.img]
        rle = _rle_from_binary_mask(p.mask.astype(np.uint8))
        coco_dt_list.append({
            "image_id": img_id,
            "category_id": int(p.cls),
            "segmentation": rle,
            "score": float(p.score),
        })

    return coco_gt, coco_dt_list, img_id_by_path, cat_id_to_name

def run_cocoeval(coco_gt: COCO, coco_dt_list: List[dict], iouType: str = "segm"):
    """Runs COCOeval once (all IoUs & areas), returns (evaluator, coco_dt)."""
    coco_dt = coco_gt.loadRes(coco_dt_list) if len(coco_dt_list) else COCO()  
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iouType)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()  
    return coco_eval, coco_dt

def extract_overall_from_stats(coco_eval: COCOeval) -> Dict[str, float]:
    s = coco_eval.stats
    return {
        "mAP_50_95": float(s[0]),
        "AP50": float(s[1]),
        "AP75": float(s[2]),
        "mAP_small": float(s[3]),
        "mAP_medium": float(s[4]),
        "mAP_large": float(s[5]),
    }

def per_class_from_precision(coco_eval: COCOeval, cat_ids: List[int]) -> Dict[int, Dict[str, float]]:
    """
    Compute class-wise AP50, AP75, and mAP[.50:.95] from the precision tensor:
    precision shape = T x R x K x A x M (see COCOeval docs) and uses -1 where undefined.
    We take area='all' (A=0) and the largest maxDet (last M). IoU slices for 0.50 / 0.75 if present.
    """
    precis = coco_eval.eval["precision"] 
    if precis is None:
        return {int(k): {"mAP_50_95": 0.0, "AP50": 0.0, "AP75": 0.0} for k in cat_ids}

    iou_thrs = coco_eval.params.iouThrs
    cat_id_to_index = {cid: i for i, cid in enumerate(coco_eval.params.catIds)}
    a_idx = 0
    m_idx = len(coco_eval.params.maxDets) - 1

    def _mean_valid(x: np.ndarray) -> float:
        x = x[x > -1]
        return float(x.mean()) if x.size else 0.0

    def _find_iou(v: float) -> Optional[int]:
        m = np.where(np.isclose(iou_thrs, v, atol=1e-6))[0]
        return int(m[0]) if m.size else None

    t50 = _find_iou(0.50)
    t75 = _find_iou(0.75)

    out = {}
    for cid in cat_ids:
        k = cat_id_to_index.get(cid, None)
        if k is None:
            out[int(cid)] = {"mAP_50_95": 0.0, "AP50": 0.0, "AP75": 0.0}
            continue
        pc = precis[:, :, k, a_idx, m_idx]  
        out_c = {"mAP_50_95": _mean_valid(pc)}
        if t50 is not None:
            out_c["AP50"] = _mean_valid(precis[t50, :, k, a_idx, m_idx][None, :])
        else:
            out_c["AP50"] = 0.0
        if t75 is not None:
            out_c["AP75"] = _mean_valid(precis[t75, :, k, a_idx, m_idx][None, :])
        else:
            out_c["AP75"] = 0.0
        out[int(cid)] = out_c
    return out

print("\n=== Evaluation===")

macro_miou = weighted_iou = mdice = None
iou_per_class = dice_per_class = None

if len(perimage_gt) == 0:
    print("No ground-truth masks found for queries. Add query masks for eval.")
else:
    CLASS_IDS = list(range(1, NUM_CLASSES+1))  
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
    print("\nStandard pixel-wise")
    for c in CLASS_IDS:
        cname = CLASS_NAMES[c-1]
        p = "n/a" if P_std[c]  is None else f"{P_std[c]:.4f}"
        r = "n/a" if R_std[c]  is None else f"{R_std[c]:.4f}"
        f = "n/a" if F1_std[c] is None else f"{F1_std[c]:.4f}"
        print(f"  class {c:>2} ({cname}): P={p} | R={r} | F1={f}")
    print(f"Macro: P={P_std_macro:.4f} | R={R_std_macro:.4f} | F1={F1_std_macro:.4f}")

    BOUNDARY_TOL_PX = 2 
    bf_overall = None
    bf_per_class = None
    bf_pool_detail = None

    bf_overall, bf_per_class, bf_pool_detail = compute_dataset_boundary_f1(
        [gt.astype(np.int32) for gt in perimage_gt],
        [pr.astype(np.int32) for pr in perimage_pred],
        CLASS_IDS,
        tol_px=BOUNDARY_TOL_PX
    )

    print(f"\nBoundary F1 @ tol={BOUNDARY_TOL_PX}px (background excluded)")
    print(f"Overall: F1={bf_pool_detail['F1']:.4f} | "
          f"P={bf_pool_detail['P']:.4f} | R={bf_pool_detail['R']:.4f} | "
          f"total_pred_boundary={bf_pool_detail['tot_pred_b']} | total_gt_boundary={bf_pool_detail['tot_gt_b']}")

    print("Per-class BF (id, name): F1 | P | R | (#pred_b | #gt_b)")
    for c in CLASS_IDS:
        stats = bf_per_class[c]
        cname = CLASS_NAMES[c - 1]
        f1  = ("n/a" if stats["F1"] is None else f"{stats['F1']:.4f}")
        p   = ("n/a" if stats["P"]  is None else f"{stats['P']:.4f}")
        r   = ("n/a" if stats["R"]  is None else f"{stats['R']:.4f}")
        print(f"  {c:>2} ({cname}): {f1} | {p} | {r} | ({stats['tot_pred_b']} | {stats['tot_gt_b']})")

if len(ALL_GTS) == 0:
    print("\n=== COCOeval (segm) ===\nNo GT instances collected → skipping COCOeval.")
    coco_metrics = None
    per_class_ap = None
else:
    coco_gt, coco_dt_list, img_id_by_path, cat_id_to_name = build_coco_gt_and_dt(
        ALL_PREDS, ALL_GTS, CLASS_NAMES, query_paths
    )
    coco_eval, coco_dt = run_cocoeval(coco_gt, coco_dt_list, iouType="segm")
    overall = extract_overall_from_stats(coco_eval)

    cat_ids = list(range(1, NUM_CLASSES + 1))
    per_class_ap = per_class_from_precision(coco_eval, cat_ids)

    print("\n=== COCOeval (segm) — key numbers ===")
    print(f"AP50: {overall['AP50']:.4f} | AP75: {overall['AP75']:.4f} | mAP[.50:.95]: {overall['mAP_50_95']:.4f}")
    print(f"mAP[.50:.95] by size → small: {overall['mAP_small']:.4f} | medium: {overall['mAP_medium']:.4f} | large: {overall['mAP_large']:.4f}")
    print("\nPer-class (id, name): AP50 | AP75 | mAP[.50:.95]")
    for cid in cat_ids:
        nm = CLASS_NAMES[cid-1]
        pc = per_class_ap.get(cid, {"AP50":0.0, "AP75":0.0, "mAP_50_95":0.0})
        print(f"  {cid:>2} ({nm}): {pc['AP50']:.4f} | {pc['AP75']:.4f} | {pc['mAP_50_95']:.4f}")

    coco_metrics = {"overall": overall, "per_class": per_class_ap}

METRICS_DIR = OUT_DIR
METRICS_PATH = METRICS_DIR / "metrics_summary.json"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

_mean = lambda v: float(np.mean(v)) if len(v) else 0.0

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

try:
    if bf_pool_detail is not None:
        summary.update({
            "boundary_f1": {
                "tolerance_px": int(BOUNDARY_TOL_PX),
                "per_class": [
                    {
                        "id": int(c),
                        "name": CLASS_NAMES[c - 1],
                        "precision": None if bf_per_class[c]["P"]  is None else round(float(bf_per_class[c]["P"]), 4),
                        "recall":    None if bf_per_class[c]["R"]  is None else round(float(bf_per_class[c]["R"]), 4),
                        "f1":        None if bf_per_class[c]["F1"] is None else round(float(bf_per_class[c]["F1"]), 4),
                        "total_pred_boundary": int(bf_per_class[c]["tot_pred_b"]),
                        "total_gt_boundary":   int(bf_per_class[c]["tot_gt_b"]),
                    }
                    for c in CLASS_IDS
                ],
                "macro": {
                    "precision": round(float(bf_pool_detail["P"]), 4),
                    "recall": round(float(bf_pool_detail["R"]), 4),
                    "f1": round(float(bf_pool_detail["F1"]), 4),
                },
            }
        })
except NameError:
    pass

if coco_metrics is not None:
    per_cls_full = []
    for c in range(1, NUM_CLASSES + 1):
        pc = coco_metrics["per_class"].get(c, None)
        per_cls_full.append({
            "id": int(c),
            "name": CLASS_NAMES[c - 1],
            "AP50": None if pc is None else round(float(pc["AP50"]), 4),
            "AP75": None if pc is None else round(float(pc["AP75"]), 4),
            "mAP_50_95": None if pc is None else round(float(pc["mAP_50_95"]), 4),
        })

    summary["instance_AP_COCO"] = {
        "AP50": round(float(coco_metrics["overall"]["AP50"]), 4),
        "AP75": round(float(coco_metrics["overall"]["AP75"]), 4),
        "mAP_50_95": round(float(coco_metrics["overall"]["mAP_50_95"]), 4),
        "mAP_50_95_by_size": {
            "small":  round(float(coco_metrics["overall"]["mAP_small"]), 4),
            "medium": round(float(coco_metrics["overall"]["mAP_medium"]), 4),
            "large":  round(float(coco_metrics["overall"]["mAP_large"]), 4),
        },
        "per_class": per_cls_full,
    }

summary = json.loads(json.dumps(summary, default=_to_py))
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"[metrics] wrote JSON to {METRICS_PATH}")

print("\n=== Timing (mean per query) ===")
print(f"TTFT (per-query) : {_mean(t_ttft):.4f} s")
if ttft_skips > 0:
    print(f"Note: TTFT skipped for {ttft_skips} queries due to no valid samples.")
print("Done")