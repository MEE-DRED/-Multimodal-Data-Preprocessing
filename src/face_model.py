"""Reusable facial-verification component (Task 2).

Public API used by the integrated application::

    from src.face_model import verify_face
    result = verify_face("path/to/image.jpg")

Result (always a dict, never raises for ordinary bad input)::

    {"verified": bool, "user_id": "member_1" | None, "score": float | None,
     "message": str,
     # extra diagnostics (safe to ignore):
     "status": "verified|below_threshold|file_not_found|unreadable_image|no_face_detected|model_unavailable|error",
     "best_match": str | None,   # nearest enrolled member EVEN IF REJECTED - never use for access control
     "threshold": float, "num_faces": int | None}

Decision rule: cosine similarity between the probe embedding and each enrolled member
template; the best match is authorised only if its similarity >= threshold. Otherwise
``verified`` is False and ``user_id`` is None, so unknown people are never forced onto
the nearest member.

This is an ACADEMIC PROTOTYPE trained on 3 originals per member. It is not suitable
for real financial authentication. This module never touches audio.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    from .feature_extraction import DEFAULT_CONFIG, FaceError, ModelUnavailable, NoFaceDetected, get_embedding
    from .image_preprocessing import load_image_rgb
    from .evaluation import cosine_scores, decide
except ImportError:  # run as a script from inside src/
    from feature_extraction import DEFAULT_CONFIG, FaceError, ModelUnavailable, NoFaceDetected, get_embedding
    from image_preprocessing import load_image_rgb
    from evaluation import cosine_scores, decide

PROJECT_ROOT = Path(__file__).resolve().parents[1]          # relative to this file, not a personal path
MODEL_DIR = PROJECT_ROOT / "models" / "face"
CONFIG_PATH = MODEL_DIR / "face_config.json"
TEMPLATES_PATH = MODEL_DIR / "face_templates.npy"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def _result(verified=False, user_id=None, score=None, message="", status="error", best_match=None,
            threshold=None, num_faces=None) -> dict:
    return {"verified": bool(verified), "user_id": user_id, "score": score, "message": message,
            "status": status, "best_match": best_match, "threshold": threshold, "num_faces": num_faces}


def default_embedder(cfg):
    """Embedder backed by the pretrained model. Returns (embedding, n_faces)."""
    def _embed(rgb):
        r = get_embedding(rgb, cfg)
        return r["embedding"], r["n_faces"]
    return _embed


class FaceVerifier:
    """Holds config + templates. ``embedder`` can be injected (used by unit tests)."""

    def __init__(self, config: dict, member_ids: list[str], templates: np.ndarray, embedder=None):
        self.cfg = {**DEFAULT_CONFIG, **config}
        self.member_ids = list(member_ids)
        self.templates = np.asarray(templates, dtype=np.float64)
        self.embedder = embedder or default_embedder(self.cfg)

    @classmethod
    def load(cls, model_dir=MODEL_DIR, embedder=None) -> "FaceVerifier":
        model_dir = Path(model_dir)
        cfg_p, tpl_p, meta_p = model_dir / "face_config.json", model_dir / "face_templates.npy", model_dir / "model_metadata.json"
        for p in (cfg_p, tpl_p, meta_p):
            if not p.exists():
                raise ModelUnavailable(f"Missing model resource: {p.name}. Run notebooks/02_image_preprocessing.ipynb to generate it.")
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        T = np.load(tpl_p, allow_pickle=False)  # plain array - no pickle
        return cls(cfg, meta["template_member_ids"], T, embedder)

    def verify(self, image_path) -> dict:
        thr = float(self.cfg["threshold"])
        try:
            path = Path(image_path)
            if not path.is_file():
                return _result(message=f"Image file not found: {path.name}", status="file_not_found", threshold=thr)
            try:
                rgb = load_image_rgb(path, self.cfg.get("max_image_side"))
            except ValueError:
                return _result(message="Image could not be read", status="unreadable_image", threshold=thr)
            try:
                emb, n_faces = self.embedder(rgb)
            except NoFaceDetected:
                return _result(message="No face detected in the image", status="no_face_detected", threshold=thr)
            except ModelUnavailable as exc:
                return _result(message=f"Face model unavailable: {exc}", status="model_unavailable", threshold=thr)
            except FaceError as exc:
                return _result(message=f"Face processing failed: {exc}", status="error", threshold=thr)
            scores = cosine_scores(np.asarray(emb), self.templates)
            d = decide(scores, self.member_ids, thr)
            if d["authorised"]:
                return _result(True, d["user_id"], round(d["score"], 4), "Face verified successfully",
                               "verified", d["best_match"], thr, n_faces)
            return _result(False, None, round(d["score"], 4), "Face verification failed",
                           "below_threshold", d["best_match"], thr, n_faces)
        except Exception as exc:  # last-resort guard: never crash the integrated app
            return _result(message=f"Unexpected error: {type(exc).__name__}", status="error", threshold=thr)


_VERIFIER: FaceVerifier | None = None


def verify_face(image_path) -> dict:
    """Integration entry point. Loads the saved model once (no notebook needed)."""
    global _VERIFIER
    if _VERIFIER is None:
        try:
            _VERIFIER = FaceVerifier.load()
        except ModelUnavailable as exc:
            return _result(message=f"Face model unavailable: {exc}", status="model_unavailable")
        except Exception as exc:
            return _result(message=f"Could not load face model: {type(exc).__name__}", status="model_unavailable")
    return _VERIFIER.verify(image_path)


def save_model_resources(member_ids, templates, config: dict, metadata: dict, model_dir=MODEL_DIR) -> None:
    """Persist templates (plain .npy) + JSON config/metadata. Used by the notebook."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(model_dir / "face_templates.npy", np.asarray(templates, dtype=np.float32), allow_pickle=False)
    (model_dir / "face_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (model_dir / "model_metadata.json").write_text(json.dumps({**metadata, "template_member_ids": list(member_ids)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m src.face_model <image_path>")
        sys.exit(2)
    print(json.dumps(verify_face(sys.argv[1]), indent=2))
