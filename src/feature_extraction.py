"""Face detection + embedding extraction using a pretrained model via DeepFace.

Primary approach: FaceNet-512 (Facenet512 in DeepFace, 160x160 input, 512-d embedding)
with the RetinaFace detector and DeepFace's built-in alignment. DeepFace is imported
lazily so that the rest of the project can be used without TensorFlow installed.

Privacy note: DeepFace downloads *model weights* from GitHub on first use. Your face
images are processed locally and are NOT uploaded anywhere by this code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .image_preprocessing import load_image_rgb, save_rgb
except ImportError:  # executed as a plain script
    from image_preprocessing import load_image_rgb, save_rgb

DEFAULT_CONFIG = {
    "model_name": "Facenet512",
    "detector_backend": "retinaface",
    "similarity_metric": "cosine",
    "threshold": 0.70,            # on cosine SIMILARITY (== cosine distance 0.30)
    "max_image_side": 1024,       # images are down-scaled to this before detection
    "align": True,
    "embedding_dim": 512,
    "template_method": "mean_of_l2_normalised_embeddings",
}
DETECTOR_CANDIDATES = ["retinaface", "mtcnn", "yunet", "opencv"]


class FaceError(Exception):
    """Base class for expected, recoverable face-pipeline problems."""


class NoFaceDetected(FaceError):
    pass


class ModelUnavailable(FaceError):
    pass


def _deepface():
    try:
        from deepface import DeepFace
        return DeepFace
    except Exception as exc:
        raise ModelUnavailable(f"DeepFace is not installed/importable: {exc}") from exc


def get_embedding(rgb: np.ndarray, cfg: dict | None = None) -> dict:
    """Detect faces in an RGB image and return the embedding of the LARGEST face.

    Returns dict(embedding, n_faces, facial_area, face_confidence).
    Raises NoFaceDetected if nothing is found (we never invent features).
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    DeepFace = _deepface()
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # DeepFace expects BGR ndarrays
    try:
        reps = DeepFace.represent(
            img_path=bgr, model_name=cfg["model_name"], detector_backend=cfg["detector_backend"],
            enforce_detection=True, align=cfg["align"],
        )
    except ValueError as exc:  # DeepFace raises ValueError when no face is found
        raise NoFaceDetected(str(exc)) from exc
    if not reps:
        raise NoFaceDetected("detector returned no faces")
    best = max(reps, key=lambda r: r["facial_area"]["w"] * r["facial_area"]["h"])
    emb = np.asarray(best["embedding"], dtype=np.float64)
    if emb.ndim != 1 or not np.isfinite(emb).all():
        raise FaceError("invalid (non-finite) embedding produced")
    return dict(embedding=emb, n_faces=len(reps), facial_area=best["facial_area"],
                face_confidence=float(best.get("face_confidence", float("nan"))))


def select_detector(image_paths, candidates=DETECTOR_CANDIDATES, base_cfg=None, max_side=1024):
    """Pick the first detector that finds >=1 face in every supplied image."""
    log = []
    for det in candidates:
        cfg = {**DEFAULT_CONFIG, **(base_cfg or {}), "detector_backend": det}
        ok, err = 0, ""
        for p in image_paths:
            try:
                get_embedding(load_image_rgb(p, max_side), cfg)
                ok += 1
            except Exception as exc:
                err = f"{type(exc).__name__}: {str(exc)[:100]}"
        log.append(dict(detector=det, detected=ok, total=len(image_paths), last_error=err))
        if ok == len(image_paths):
            return det, pd.DataFrame(log)
    return None, pd.DataFrame(log)


def save_aligned_crop(rgb, out_path, cfg=None, size=(160, 160)):
    """Save the aligned, resized face crop (documentation/visual QA of preprocessing)."""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    DeepFace = _deepface()
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    faces = DeepFace.extract_faces(img_path=bgr, detector_backend=cfg["detector_backend"],
                                   enforce_detection=True, align=cfg["align"])
    f = max(faces, key=lambda r: r["facial_area"]["w"] * r["facial_area"]["h"])
    import cv2
    face = f["face"]
    face = (face * 255).clip(0, 255).astype(np.uint8) if face.dtype != np.uint8 else face
    face = cv2.resize(face, size, interpolation=cv2.INTER_AREA)  # RGB (DeepFace returns RGB)
    save_rgb(face, out_path)
    return f["facial_area"]


def build_feature_table(originals_df: pd.DataFrame, aug_manifest: pd.DataFrame, cfg: dict | None = None):
    """Embeddings for every original + augmented image.

    Returns (features_df, failures_df). Failed detections go to failures_df and are
    NEVER filled with made-up values.
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    jobs = [dict(member_id=r.member_id, image_filename=r.image_filename, expression=r.expression,
                 augmentation_type="none", original_filename=r.image_filename, is_original=True, path=r.path)
            for r in originals_df[originals_df["readable"]].itertuples()]
    jobs += [dict(member_id=r.member_id, image_filename=r.output_filename, expression=r.expression,
                  augmentation_type=r.augmentation_type, original_filename=r.original_filename,
                  is_original=False, path=r.output_path) for r in aug_manifest.itertuples()]
    rows, fails = [], []
    for j in jobs:
        path = j.pop("path")
        try:
            res = get_embedding(load_image_rgb(path, cfg["max_image_side"]), cfg)
            fa = res["facial_area"]
            rec = {**j, "detector_backend": cfg["detector_backend"], "model_name": cfg["model_name"],
                   "n_faces_detected": res["n_faces"], "face_confidence": res["face_confidence"],
                   "face_x": fa["x"], "face_y": fa["y"], "face_w": fa["w"], "face_h": fa["h"]}
            rec.update({f"embedding_{i}": float(v) for i, v in enumerate(res["embedding"])})
            rows.append(rec)
        except Exception as exc:
            fails.append({**j, "path": path, "error_type": type(exc).__name__, "error": str(exc)[:200]})
    return pd.DataFrame(rows), pd.DataFrame(fails)


def embedding_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("embedding_")]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))
