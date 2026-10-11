"""Evaluation metrics (Appendix A.5), all between one reconstruction and one
target photograph, over the full image:

    sscd     SSCD cosine similarity (reported as AS; A-ESR thresholds it)
    lpips    LPIPS, AlexNet backbone, 512x512
    ssim     SSIM over RGB channels, 512x512
    clip     CLIP ViT-B/32 image-embedding cosine similarity
    arcface  ArcFace cosine similarity of the detected faces, None when no face
             is detected in either image (faces only)
"""
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
from torchvision import transforms

import config as C

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SSCD_TRANSFORM = transforms.Compose([      # FineXtract's preprocessing
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
])

_models = {}


def _sscd():
    if "sscd" not in _models:
        _models["sscd"] = torch.jit.load(C.WEIGHTS["sscd"], map_location=DEVICE).eval()
    return _models["sscd"]


def _lpips():
    if "lpips" not in _models:
        import lpips
        _models["lpips"] = lpips.LPIPS(net="alex", verbose=False).to(DEVICE)
    return _models["lpips"]


def _clip():
    if "clip" not in _models:
        import clip
        _models["clip"] = clip.load("ViT-B/32", device=DEVICE, jit=False)
    return _models["clip"]


def _faces():
    if "faces" not in _models:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(320, 320))
        _models["faces"] = app
    return _models["faces"]


# ---------------------------------------------------------------- features --
@torch.no_grad()
def sscd_feature(path):
    x = SSCD_TRANSFORM(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    return _sscd()(x).flatten().cpu()


def sscd_features(paths):
    return torch.stack([sscd_feature(p) for p in paths])


@torch.no_grad()
def clip_feature(path):
    model, preprocess = _clip()
    x = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    f = model.encode_image(x)
    return (f / f.norm(dim=-1, keepdim=True)).flatten().cpu()


def face_embedding(path):
    """512-d ArcFace embedding of the most confident detected face, or None."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    faces = _faces().get(img)
    if not faces:
        return None
    best = max(faces, key=lambda f: f.det_score)
    return torch.from_numpy(best.embedding.copy())


@lru_cache(maxsize=256)
def _lpips_tensor(path, size=512):
    arr = np.asarray(Image.open(path).convert("RGB").resize((size, size))).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


@lru_cache(maxsize=256)
def _rgb_array(path, size=512):
    return np.asarray(Image.open(path).convert("RGB").resize((size, size)))


# ----------------------------------------------------------------- scoring --
def cosine(a, b):
    return F.cosine_similarity(a.flatten().unsqueeze(0).float(), b.flatten().unsqueeze(0).float()).item()


@torch.no_grad()
def score_pair(gen_path, target_path, faces=True):
    """All metrics between one reconstruction and its target."""
    row = {
        "sscd": cosine(sscd_feature(gen_path), sscd_feature(target_path)),
        "lpips": _lpips()(_lpips_tensor(gen_path), _lpips_tensor(target_path)).item(),
        "ssim": float(structural_similarity(_rgb_array(gen_path), _rgb_array(target_path),
                                            channel_axis=-1, data_range=255)),
        "clip": cosine(clip_feature(gen_path), clip_feature(target_path)),
        "arcface": None,
    }
    if faces:
        g, t = face_embedding(gen_path), face_embedding(target_path)
        if g is not None and t is not None:
            row["arcface"] = cosine(g, t)
    return row


@torch.no_grad()
def score_pool(gen_paths, real_paths, faces=True):
    """Free-form evaluation (Section 5.1): each real photograph is matched,
    separately for each metric, to the extracted image yielding the best score.
    AS is the highest SSCD similarity, LPIPS the lowest distance, and SSIM, CLIP
    and ArcFace the highest similarity over the extracted images. One row per
    real photograph; `matched` names the SSCD-best image."""
    rows = []
    for rp in real_paths:
        scored = {g: score_pair(g, rp, faces=faces) for g in gen_paths}
        best = max(scored, key=lambda g: scored[g]["sscd"])
        faces_found = [s["arcface"] for s in scored.values() if s["arcface"] is not None]
        rows.append({
            "sscd": scored[best]["sscd"],
            "lpips": min(s["lpips"] for s in scored.values()),
            "ssim": max(s["ssim"] for s in scored.values()),
            "clip": max(s["clip"] for s in scored.values()),
            "arcface": max(faces_found) if faces_found else None,
            "photo": rp, "matched": best,
        })
    return rows
