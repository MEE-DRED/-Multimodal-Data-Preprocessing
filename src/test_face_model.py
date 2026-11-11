"""Tests for the face verifier.  Run from the project root:

    python -m unittest src.test_face_model -v

Two kinds of test:
  [LOGIC]  use an injected stand-in embedder with synthetic vectors. They check the
           decision/failure-handling code only. They say NOTHING about model accuracy.
  [MODEL]  use the real pretrained model + saved templates. They are SKIPPED (not
           passed) when DeepFace / model resources / consented test images are absent.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.face_model import FaceVerifier, MODEL_DIR, PROJECT_ROOT, verify_face
from src.feature_extraction import NoFaceDetected
from src.evaluation import l2_normalize

RAW = PROJECT_ROOT / "data" / "raw" / "images"
UNKNOWN_DIR = PROJECT_ROOT / "data" / "test_unknown"


def _unit(i, dim=16):
    v = np.zeros(dim); v[i] = 1.0
    return v


def _verifier(embed_fn, thr=0.7):
    T = np.vstack([_unit(0), _unit(1), _unit(2)])
    return FaceVerifier({"threshold": thr}, ["member_1", "member_2", "member_3"], T, embedder=embed_fn)


def _make_valid_jpg(d):
    from PIL import Image
    p = Path(d) / "x.jpg"
    Image.fromarray(np.full((64, 64, 3), 127, np.uint8)).save(p)
    return p


def _model_ready():
    try:
        import deepface  # noqa: F401
    except Exception:
        return False
    return all((MODEL_DIR / n).exists() for n in ("face_config.json", "face_templates.npy", "model_metadata.json"))


class LogicTests(unittest.TestCase):
    def test_logic_enrolled_member_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            r = _verifier(lambda rgb: (_unit(1) + 0.05 * _unit(5), 1)).verify(_make_valid_jpg(d))
        self.assertTrue(r["verified"]); self.assertEqual(r["user_id"], "member_2"); self.assertGreaterEqual(r["score"], 0.7)

    def test_logic_missing_file(self):
        r = _verifier(lambda rgb: (_unit(0), 1)).verify("does/not/exist.jpg")
        self.assertFalse(r["verified"]); self.assertEqual(r["status"], "file_not_found"); self.assertIsNone(r["user_id"])

    def test_logic_unreadable_image(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "broken.jpg"; p.write_text("this is not an image")
            r = _verifier(lambda rgb: (_unit(0), 1)).verify(p)
        self.assertFalse(r["verified"]); self.assertEqual(r["status"], "unreadable_image")

    def test_logic_no_face_detected(self):
        def no_face(rgb): raise NoFaceDetected("none")
        with tempfile.TemporaryDirectory() as d:
            r = _verifier(no_face).verify(_make_valid_jpg(d))
        self.assertFalse(r["verified"]); self.assertEqual(r["status"], "no_face_detected")

    def test_logic_insufficient_similarity_rejected_not_forced_to_nearest(self):
        # similarity to best template = 0.4 < 0.7 -> must be rejected, user_id None
        v = l2_normalize(0.4 * _unit(0) + np.sqrt(1 - 0.16) * _unit(9))
        with tempfile.TemporaryDirectory() as d:
            r = _verifier(lambda rgb: (v, 1)).verify(_make_valid_jpg(d))
        self.assertFalse(r["verified"]); self.assertIsNone(r["user_id"])
        self.assertEqual(r["status"], "below_threshold"); self.assertEqual(r["best_match"], "member_1")
        self.assertAlmostEqual(r["score"], 0.4, places=3)

    def test_logic_unexpected_embedder_error_does_not_crash(self):
        def boom(rgb): raise RuntimeError("boom")
        with tempfile.TemporaryDirectory() as d:
            r = _verifier(boom).verify(_make_valid_jpg(d))
        self.assertFalse(r["verified"]); self.assertEqual(r["status"], "error")

    def test_logic_result_schema(self):
        r = verify_face("does/not/exist.jpg")
        for k in ("verified", "user_id", "score", "message"):
            self.assertIn(k, r)


@unittest.skipUnless(_model_ready(), "[MODEL] DeepFace and/or saved model resources not available - run the notebook first")
class ModelTests(unittest.TestCase):
    def test_model_enrolled_images_verify(self):
        # NOTE: these images were used to build the templates -> this is a smoke test, NOT an accuracy estimate.
        for m in ("member_1", "member_2", "member_3"):
            r = verify_face(RAW / m / f"{m}_neutral.jpg")
            self.assertTrue(r["verified"], r); self.assertEqual(r["user_id"], m)

    def test_model_image_without_face(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "blank.jpg"
            Image.fromarray(np.full((300, 300, 3), 200, np.uint8)).save(p)
            r = verify_face(p)
        self.assertFalse(r["verified"]); self.assertEqual(r["status"], "no_face_detected")

    def test_model_unknown_person_rejected(self):
        imgs = [p for p in UNKNOWN_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not imgs:
            self.skipTest("PENDING: no consented unknown-person image in data/test_unknown/")
        for p in imgs:
            r = verify_face(p)
            self.assertFalse(r["verified"], f"{p.name} was accepted: {r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
