"""Tests for unknown-person testing logic. Run from the project root:

    python -m unittest src.test_unknown_testing -v

[LOGIC] tests use a stand-in embedder with synthetic vectors: they check decision/failure/metric
code only and say NOTHING about the accuracy of the real FaceNet512 model.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src import unknown_testing as ut
from src.feature_extraction import ModelUnavailable, NoFaceDetected

IDS = ["member_1", "member_2", "member_3"]
T = np.eye(3, 16)            # three orthonormal templates
THR = 0.7


def _img(d, name="x.jpg"):
    p = Path(d) / name
    Image.fromarray(np.full((64, 64, 3), 127, np.uint8)).save(p)
    return p


def _emb(vec):
    return lambda rgb: dict(embedding=np.asarray(vec, float), n_faces=1, face_confidence=0.99)


def _row(status="processed", decision="REJECTED", score=0.3, person="person_1", best="member_1"):
    return dict(person_id=person, image_filename="a.jpg", status=status, decision=decision, best_score=score, best_match=best)


class LogicTests(unittest.TestCase):
    def test_below_threshold_is_rejected_even_with_closest_member(self):
        with tempfile.TemporaryDirectory() as d:
            v = np.zeros(16); v[0] = 0.3; v[5] = 1.0
            r = ut.evaluate_unknown_image(_img(d), _emb(v), T, IDS, THR)
        self.assertEqual(r["decision"], "REJECTED"); self.assertEqual(r["outcome"], "CORRECT REJECTION")
        self.assertEqual(r["best_match"], "member_1"); self.assertLess(r["best_score"], THR)

    def test_at_or_above_threshold_is_false_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            r = ut.evaluate_unknown_image(_img(d), _emb(np.eye(16)[1]), T, IDS, THR)
            self.assertEqual(r["decision"], "ACCEPTED"); self.assertEqual(r["outcome"], "FALSE ACCEPTANCE")
            r2 = ut.evaluate_unknown_image(_img(d), _emb(np.eye(16)[1]), T, IDS, 1.0)   # score == threshold
        self.assertEqual(r2["decision"], "ACCEPTED")

    def test_failures_are_reported_not_counted_as_rejections(self):
        with tempfile.TemporaryDirectory() as d:
            p = _img(d)
            def boom(exc):
                def f(rgb): raise exc
                return f
            cases = [(boom(NoFaceDetected("none")), "no_face_detected"), (boom(ModelUnavailable("x")), "model_unavailable"),
                     (boom(RuntimeError("oops")), "embedding_failed"), (_emb(np.full(16, np.nan)), "embedding_failed"),
                     (_emb(np.ones(8)), "embedding_failed")]
            for fn, status in cases:
                r = ut.evaluate_unknown_image(p, fn, T, IDS, THR)
                self.assertEqual(r["status"], status); self.assertEqual(r["decision"], ut.DECISION_FAILED)
                self.assertEqual(r["outcome"], ut.OUTCOME_FAILED)
            self.assertEqual(ut.evaluate_unknown_image(Path(d) / "nope.jpg", _emb(np.eye(16)[0]), T, IDS, THR)["status"], "file_missing")
            bad = Path(d) / "bad.jpg"; bad.write_text("not an image")
            self.assertEqual(ut.evaluate_unknown_image(bad, _emb(np.eye(16)[0]), T, IDS, THR)["status"], "unreadable_image")

    def test_templates_not_modified(self):
        T0 = T.copy()
        with tempfile.TemporaryDirectory() as d:
            ut.evaluate_unknown_image(_img(d), _emb(np.eye(16)[2]), T, IDS, THR)
        self.assertTrue(np.array_equal(T, T0))

    def test_metrics_exclude_failures_and_handle_empty(self):
        df = pd.DataFrame([_row(), _row(), _row(decision="ACCEPTED", score=0.8), _row(status="no_face_detected", decision=ut.DECISION_FAILED, score=None)])
        m = ut.unknown_person_metrics(df, THR)
        self.assertEqual((m["n_unknown_images"], m["n_processed"], m["n_failed"], m["correct_rejections"], m["false_acceptances"]), (4, 3, 1, 2, 1))
        self.assertAlmostEqual(m["FAR"], 1 / 3)
        e = ut.unknown_person_metrics(df.iloc[0:0], THR)
        self.assertIsNone(e["FAR"]); self.assertEqual(e["n_processed"], 0)

    def test_binary_metrics(self):
        m = ut.binary_metrics(tp=9, fn=0, fp=1, tn=17)
        self.assertAlmostEqual(m["precision"], 0.9); self.assertEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["FAR"], 1 / 18); self.assertEqual(m["FRR"], 0.0)
        self.assertAlmostEqual(m["accuracy"], 26 / 27)
        z = ut.binary_metrics(0, 0, 0, 0); self.assertIsNone(z["accuracy"]); self.assertIsNone(z["f1"])

    def test_combined_evaluation_counts(self):
        loeo = pd.DataFrame(dict(true_id=["member_1", "member_2", "member_3"], user_id=["member_1", "member_2", None],
                                 best_match=["member_1", "member_2", "member_3"], score=[.9, .9, .6]))
        lomo = pd.DataFrame(dict(true_id=["unknown"] * 2, user_id=[None, None], best_match=["member_2", "member_1"], score=[.4, .5]))
        res = pd.DataFrame([_row(), _row(decision="ACCEPTED", score=0.8, best="member_2"), _row(status="no_face_detected", decision=ut.DECISION_FAILED, score=None)])
        out = ut.combined_evaluation(dict(loeo=loeo, unknown=lomo), res, THR, IDS)
        self.assertEqual((out["initial"]["TP"], out["initial"]["FN"], out["initial"]["FP"], out["initial"]["TN"]), (2, 1, 0, 2))
        self.assertEqual((out["combined"]["TP"], out["combined"]["FN"], out["combined"]["FP"], out["combined"]["TN"]), (2, 1, 1, 3))
        self.assertEqual(out["combined"]["n"], out["initial"]["n"] + 2)           # failed image excluded
        cm = np.array(out["combined_four_class"]["confusion_matrix"])
        self.assertEqual(cm.sum(), 3 + 2 + 2); self.assertEqual(cm[3, 1], 1)       # unknown accepted as member_2

    def test_parse_names_with_typos(self):
        P = ut.parse_person_and_expression
        self.assertEqual(P("Unauthorized-persons/Person 2/suprised.jpeg"), ("person_2", "surprised"))
        self.assertEqual(P("images/Unauthorized-persons/Person 1/surprised.jepg.jpeg"), ("person_1", "surprised"))
        self.assertEqual(P("Person 3/smiling.jpeg"), ("person_3", "smiling"))
        self.assertEqual(P("Member 1/neutral.jpg")[0], None)


if __name__ == "__main__":
    unittest.main()
