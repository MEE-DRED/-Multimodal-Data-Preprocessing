"""Unknown-person (unauthorised access) testing for the face verifier.

Pure NumPy / pandas / Pillow / scikit-learn: no DeepFace import here, so the logic can be
unit-tested without TensorFlow. The face model is injected as ``embed_fn`` (the notebook
passes ``fx.get_embedding``).

Design rules (see docs/evaluation_report.md):

* Unknown-person images live in ``data/test_unknown/person_N/`` and are NEVER written to
  ``data/raw/images`` or to the enrolled feature table, and never used to build or update
  templates. Templates are only READ here (and the notebook hashes them before/after).
* The decision is made by the fixed threshold, not by "closest member": an unknown image is
  ACCEPTED only if its best cosine similarity is >= the threshold (a false acceptance).
* A failure (missing file, unreadable file, no face, model unavailable, bad embedding) is
  reported as a failure. It is never counted as a correct rejection.
* Nothing is invented: any metric whose denominator is zero is returned as ``None``.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .evaluation import cosine_scores, decide, open_set_classification, wilson_interval, UNKNOWN
    from .feature_extraction import ModelUnavailable, NoFaceDetected
    from .image_preprocessing import EXPRESSIONS, IMAGE_EXTS, load_image_rgb, sha256_of
except ImportError:  # executed as a plain script from inside src/
    from evaluation import cosine_scores, decide, open_set_classification, wilson_interval, UNKNOWN
    from feature_extraction import ModelUnavailable, NoFaceDetected
    from image_preprocessing import EXPRESSIONS, IMAGE_EXTS, load_image_rgb, sha256_of

STATUS_OK = "processed"
DECISION_ACCEPTED = "ACCEPTED"
DECISION_REJECTED = "REJECTED"
DECISION_FAILED = "NOT EVALUATED (failure)"
OUTCOME_CORRECT_REJECTION = "CORRECT REJECTION"
OUTCOME_FALSE_ACCEPTANCE = "FALSE ACCEPTANCE"
OUTCOME_FAILED = "FAILED - not counted as a rejection"


# ----------------------------------------------------------------- naming / ingestion
def parse_person_and_expression(rel_path: str):
    """Infer (person_id, expression) from e.g. 'Person 2/suprised.jpeg'.

    Tolerates the typos found in the supplied files ('suprised', '.jepg'). Returns None for any
    field that cannot be determined (never guesses).
    """
    p = str(rel_path).lower().replace("\\", "/")
    m = re.search(r"person[\s_\-]*([0-9]+)", p)
    person = f"person_{int(m.group(1))}" if m else None
    expr = None
    if "neutral" in p:
        expr = "neutral"
    elif "smil" in p:
        expr = "smiling"
    elif re.search(r"su+r*p+r*i+s", p):          # 'surprised' and the typo 'suprised'
        expr = "surprised"
    return person, expr


def enrolled_hashes(enrolled_dir) -> dict:
    """sha256 -> path for every image under the enrolled-member directory."""
    enrolled_dir = Path(enrolled_dir)
    if not enrolled_dir.exists():
        return {}
    return {sha256_of(p): str(p) for p in enrolled_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def ingest_unknown_zip(zip_path, unknown_dir, mapping_csv=None, enrolled_dir=None) -> pd.DataFrame:
    """Copy unknown-person images from a ZIP into ``unknown_dir/person_N/person_N_<expression>.jpg``.

    * Files are copied byte-for-byte; only the name changes.
    * Files byte-identical to an ENROLLED image (e.g. a 'Member N' folder that was zipped
      together with the unknown persons) are NOT copied and are listed as skipped, so enrolled
      photos can never end up in the unknown test set.
    * Every file in the ZIP appears in the returned mapping with a status; nothing is dropped silently.
    """
    zip_path, unknown_dir = Path(zip_path), Path(unknown_dir)
    tmp = unknown_dir.parent / "_unknown_zip_extract_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    known = enrolled_hashes(enrolled_dir) if enrolled_dir else {}
    rows = []
    for f in sorted(p for p in tmp.rglob("*") if p.is_file() and "__MACOSX" not in p.parts):
        rel = str(f.relative_to(tmp))
        row = {"zip_relative_path": rel, "original_extension": "".join(f.suffixes).lower(),
               "person_id": None, "expression": None, "canonical_path": None}
        if f.suffix.lower() not in IMAGE_EXTS:
            row["status"] = "skipped_not_an_image"
        elif sha256_of(f) in known:
            row["status"] = "skipped_identical_to_enrolled_image"
        else:
            person, expr = parse_person_and_expression(rel)
            row.update(person_id=person, expression=expr)
            if person is None or expr is None:
                row["status"] = "unrecognised_person_or_expression"
            else:
                dest = unknown_dir / person / f"{person}_{expr}.jpg"
                if dest.exists():
                    row["status"] = "duplicate_target_name"
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(f, dest)
                    row.update(status="ok", canonical_path=f"{person}/{dest.name}")
        rows.append(row)
    shutil.rmtree(tmp, ignore_errors=True)
    df = pd.DataFrame(rows)
    if mapping_csv:
        Path(mapping_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(mapping_csv, index=False)
    return df


# ----------------------------------------------------------------- inspection
def inspect_unknown_images(unknown_dir, enrolled_dir=None) -> pd.DataFrame:
    """One row per image found under ``unknown_dir`` (readability, size, format, leakage check)."""
    from PIL import Image  # local import keeps module import light

    unknown_dir = Path(unknown_dir)
    known = enrolled_hashes(enrolled_dir) if enrolled_dir else {}
    rows = []
    files = sorted(p for p in unknown_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if unknown_dir.exists() else []
    for f in files:
        person, expr = parse_person_and_expression(f.stem)
        person = person or (f.parent.name if f.parent.name.startswith("person_") else None)
        sha = sha256_of(f)
        row = dict(person_id=person, expression=expr, image_filename=f.name, path=str(f),
                   file_size_kb=round(f.stat().st_size / 1024, 1), sha256=sha,
                   identical_to_enrolled_image=known.get(sha))
        try:
            with Image.open(f) as im:
                row.update(format=im.format, width=im.width, height=im.height, mode=im.mode,
                           exif_orientation=im.getexif().get(274, 1))
                im.load()
            row.update(readable=True, error="")
        except Exception as exc:
            row.update(readable=False, error=str(exc))
        rows.append(row)
    return pd.DataFrame(rows)


def missing_unknown_items(df: pd.DataFrame, expressions=EXPRESSIONS) -> list:
    """(person, expression) pairs missing for the persons that ARE present."""
    if df.empty:
        return []
    have = set(zip(df["person_id"], df["expression"]))
    return [(p, e) for p in sorted(df["person_id"].dropna().unique()) for e in expressions if (p, e) not in have]


# ----------------------------------------------------------------- per-image decision
def evaluate_unknown_image(path, embed_fn, templates, member_ids, threshold, max_side=1024) -> dict:
    """Run one unknown-person image through the verification pipeline.

    ``embed_fn(rgb)`` must return a dict with ``embedding`` (and optionally ``n_faces``,
    ``face_confidence``, ``facial_area``), like ``feature_extraction.get_embedding``.
    The ground truth for every image handled here is 'not enrolled', so:
    best score >= threshold -> ACCEPTED -> FALSE ACCEPTANCE;  otherwise REJECTED -> CORRECT REJECTION.
    """
    templates = np.asarray(templates, dtype=np.float64)
    out = dict(status="", error="", face_detected=False, n_faces=None, face_confidence=None,
               face_x=None, face_y=None, face_w=None, face_h=None,
               best_match=None, best_score=None, decision=DECISION_FAILED, outcome=OUTCOME_FAILED,
               **{f"score_{m}": None for m in member_ids})

    def fail(status, msg=""):
        out.update(status=status, error=str(msg)[:200])
        return out

    path = Path(path)
    if not path.is_file():
        return fail("file_missing", f"File not found: {path.name}")
    try:
        rgb = load_image_rgb(path, max_side)                  # EXIF fix + RGB + size cap
    except ValueError as exc:
        return fail("unreadable_image", exc)
    try:
        res = embed_fn(rgb)                                   # RetinaFace + alignment + 160x160 + FaceNet512
    except NoFaceDetected as exc:
        return fail("no_face_detected", exc)
    except ModelUnavailable as exc:
        return fail("model_unavailable", exc)
    except Exception as exc:                                  # any other extraction problem is still reported
        return fail("embedding_failed", f"{type(exc).__name__}: {exc}")

    out.update(face_detected=True, n_faces=res.get("n_faces"), face_confidence=res.get("face_confidence"))
    fa = res.get("facial_area") or {}
    out.update(face_x=fa.get("x"), face_y=fa.get("y"), face_w=fa.get("w"), face_h=fa.get("h"))
    emb = np.asarray(res.get("embedding"), dtype=np.float64)
    if emb.ndim != 1 or emb.shape[0] != templates.shape[1] or not np.isfinite(emb).all():
        out["face_detected"] = True
        return fail("embedding_failed", f"invalid embedding (shape {emb.shape}, expected ({templates.shape[1]},) and finite)")

    scores = cosine_scores(emb, templates)
    d = decide(scores, list(member_ids), float(threshold))
    accepted = bool(d["authorised"])
    out.update(status=STATUS_OK, best_match=d["best_match"], best_score=float(d["score"]),
               decision=DECISION_ACCEPTED if accepted else DECISION_REJECTED,
               outcome=OUTCOME_FALSE_ACCEPTANCE if accepted else OUTCOME_CORRECT_REJECTION)
    out.update({f"score_{m}": float(s) for m, s in zip(member_ids, scores)})
    return out


def detection_status_text(row) -> str:
    if row["status"] == STATUS_OK:
        conf = row.get("face_confidence")
        conf_txt = "" if conf is None or pd.isna(conf) else f", confidence {float(conf):.2f}"
        n = row.get("n_faces")
        faces = "" if n is None or pd.isna(n) else f"{int(n)} face{'s' if int(n) != 1 else ''}"
        extra = " (largest used)" if n is not None and not pd.isna(n) and int(n) > 1 else ""
        return f"Detected ({faces}{conf_txt}){extra}"
    if row["status"] == "embedding_failed" and row.get("face_detected"):
        return "Detected, but embedding FAILED"
    return f"FAILED: {row['status']}"


# ----------------------------------------------------------------- metrics
def per_person_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in results.groupby("person_id", dropna=False):
        ok = g[g["status"] == STATUS_OK]
        top = ok.loc[ok["best_score"].idxmax()] if len(ok) else None
        rows.append(dict(person_id=pid, n_images=len(g), n_processed=len(ok),
                         n_failed=len(g) - len(ok),
                         correct_rejections=int((ok["decision"] == DECISION_REJECTED).sum()),
                         false_acceptances=int((ok["decision"] == DECISION_ACCEPTED).sum()),
                         highest_score=None if top is None else round(float(top["best_score"]), 4),
                         closest_identity_at_highest=None if top is None else top["best_match"]))
    return pd.DataFrame(rows)


def unknown_person_metrics(results: pd.DataFrame, threshold: float) -> dict:
    """Metrics for the unknown-person test alone (every processed image should be rejected)."""
    n_total = int(len(results))
    ok = results[results["status"] == STATUS_OK] if n_total else results
    n_proc = int(len(ok))
    fa = int((ok["decision"] == DECISION_ACCEPTED).sum()) if n_proc else 0
    cr = int((ok["decision"] == DECISION_REJECTED).sum()) if n_proc else 0
    scores = ok["best_score"].astype(float).to_numpy() if n_proc else np.array([])
    return dict(
        threshold=float(threshold), n_unknown_images=n_total, n_processed=n_proc, n_failed=n_total - n_proc,
        correct_rejections=cr, false_acceptances=fa,
        FAR=(fa / n_proc) if n_proc else None,
        FAR_ci95=wilson_interval(fa, n_proc) if n_proc else (None, None),
        correct_rejection_rate=(cr / n_proc) if n_proc else None,
        max_best_score=float(scores.max()) if n_proc else None,
        mean_best_score=float(scores.mean()) if n_proc else None,
        min_best_score=float(scores.min()) if n_proc else None,
        margin_below_threshold=(float(threshold) - float(scores.max())) if n_proc else None,
    )


def binary_metrics(tp: int, fn: int, fp: int, tn: int) -> dict:
    """Authorised-vs-unauthorised metrics. Positive class = genuine enrolled attempt that should be accepted."""
    div = lambda a, b: (a / b) if b else None
    n = tp + fn + fp + tn
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * precision * recall, precision + recall) if precision is not None and recall is not None and (precision + recall) else None
    return dict(n=n, TP=tp, FN=fn, FP=fp, TN=tn, accuracy=div(tp + tn, n), precision=precision, recall=recall,
                f1=f1, FAR=div(fp, fp + tn), FRR=div(fn, tp + fn))


def new_unknown_as_open_set_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Processed unknown images in the same layout as the evaluation module's unknown trials."""
    ok = results[results["status"] == STATUS_OK]
    accepted = ok["decision"] == DECISION_ACCEPTED
    user = pd.Series([b if a else None for b, a in zip(ok["best_match"], accepted)], index=ok.index, dtype=object)
    return pd.DataFrame(dict(protocol="unknown_person_test", probe_file=ok["image_filename"].to_numpy(), true_id=UNKNOWN,
                             best_match=ok["best_match"].to_numpy(), score=ok["best_score"].astype(float).to_numpy(),
                             authorised=accepted.to_numpy(), user_id=user.to_numpy()))


def combined_evaluation(R: dict, results: pd.DataFrame, threshold: float, members) -> dict:
    """Initial (enrolled-member) vs unknown-person vs combined metrics. All inputs are real outputs.

    Initial  = the notebook's open-set samples: 9 held-out genuine probes (leave-one-expression-out)
               + 9 leave-one-member-out 'unknown' probes (enrolled members treated as not enrolled).
    Unknown  = the real, never-enrolled persons tested against the final templates.
    Combined = Initial + Unknown. No new genuine attempts are added, so FRR is carried over unchanged.
    A genuine probe only counts as correctly accepted if it is accepted AS ITS OWN identity.
    """
    loeo, lomo = R["loeo"].copy(), R["unknown"].copy()
    for df in (loeo, lomo):
        df["user_id"] = df["user_id"].astype(object).where(df["user_id"].notna(), None)
    tp = int(sum(1 for u, t in zip(loeo["user_id"], loeo["true_id"]) if u is not None and u == t))
    fn = int(len(loeo) - tp)
    fp_i = int(sum(1 for u in lomo["user_id"] if u is not None))
    tn_i = int(len(lomo) - fp_i)

    um = unknown_person_metrics(results, threshold)
    fp_u, tn_u = um["false_acceptances"], um["correct_rejections"]

    initial = binary_metrics(tp, fn, fp_i, tn_i)
    unknown_only = dict(n=um["n_processed"], correct_rejections=tn_u, false_acceptances=fp_u,
                        accuracy=um["correct_rejection_rate"], FAR=um["FAR"],
                        precision=None, recall=None, f1=None, FRR=None)   # no genuine attempts in this set
    combined = binary_metrics(tp, fn, fp_i + fp_u, tn_i + tn_u)

    new_rows = new_unknown_as_open_set_rows(results)
    four = None
    if len(new_rows):
        unk_all = pd.concat([lomo, new_rows], ignore_index=True)
        # concat can turn None into NaN; open_set_classification treats only None as 'rejected'
        unk_all["user_id"] = unk_all["user_id"].astype(object).where(unk_all["user_id"].notna(), None)
        four = open_set_classification(loeo, unk_all, threshold, list(members))
    return dict(initial=initial, unknown_only=unknown_only, combined=combined, combined_four_class=four,
                unknown_metrics=um)
