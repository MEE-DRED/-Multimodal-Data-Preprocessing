"""Template building and leakage-safe evaluation for the face verifier.

Pure NumPy / pandas / scikit-learn (no DeepFace needed). Rules enforced here:

* Partitioning is done by ORIGINAL image/expression. An augmented copy always follows
  its original, so an original and its own augmentations are never split across
  enrolment and test.
* Probes are never part of the templates used to score them.
* No samples are invented. If a metric cannot be computed it is returned as None.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)

try:
    from .feature_extraction import embedding_columns
except ImportError:
    from feature_extraction import embedding_columns

UNKNOWN = "unknown"


# ----------------------------------------------------------------- basics
def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1, n)


def make_template(embeddings: np.ndarray) -> np.ndarray:
    """Mean of L2-normalised embeddings, re-normalised (unit-length template)."""
    return l2_normalize(l2_normalize(embeddings).mean(axis=0))


def build_templates(df: pd.DataFrame, cols: list[str], member_ids: list[str] | None = None):
    """Return (member_ids, templates[n_members, dim]) from an enrolment DataFrame."""
    member_ids = member_ids or sorted(df["member_id"].unique())
    T = np.vstack([make_template(df.loc[df["member_id"] == m, cols].to_numpy()) for m in member_ids])
    return list(member_ids), T


def cosine_scores(emb: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Cosine similarity of one embedding to every template (higher = more similar)."""
    return templates @ l2_normalize(emb)


def decide(scores: np.ndarray, member_ids: list[str], threshold: float) -> dict:
    """Separate IDENTITY MATCHING (best candidate) from the AUTHORISATION decision."""
    i = int(np.argmax(scores))
    best_id, best = member_ids[i], float(scores[i])
    authorised = best >= threshold
    return dict(best_match=best_id, score=best, authorised=authorised,
                user_id=best_id if authorised else None)


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - a) / d, (c + a) / d)


# -------------------------------------------------------------- protocols
def leave_one_expression_out(features: pd.DataFrame, threshold: float, use_aug_in_enrolment: bool = True,
                             members=None) -> pd.DataFrame:
    """Closed/verification protocol. For each held-out expression e:

    enrol on every other expression (originals + their augmentations),
    probe with the held-out ORIGINAL of each member (independent of enrolment).
    Each probe is scored against all member templates.
    """
    cols = embedding_columns(features)
    members = members or sorted(features["member_id"].unique())
    rows = []
    for e in sorted(features["expression"].unique()):
        enrol = features[(features["expression"] != e) & (features["member_id"].isin(members))]
        if not use_aug_in_enrolment:
            enrol = enrol[enrol["is_original"]]
        ids, T = build_templates(enrol, cols, members)
        probes = features[(features["expression"] == e) & features["is_original"] & features["member_id"].isin(members)]
        for r in probes.itertuples(index=False):
            s = cosine_scores(np.array([getattr(r, c) for c in cols]), T)
            d = decide(s, ids, threshold)
            rows.append(dict(protocol="leave_one_expression_out", probe_file=r.image_filename, true_id=r.member_id,
                             held_out_expression=e, **{f"score_{m}": float(v) for m, v in zip(ids, s)}, **d))
    return pd.DataFrame(rows)


def unknown_person_trials(features: pd.DataFrame, threshold: float, use_aug_in_enrolment: bool = True) -> pd.DataFrame:
    """Open-set protocol using REAL data: treat one enrolled member as 'unknown'.

    For each member u, enrol the OTHER members only (all their images) and probe with
    u's original images. u was never enrolled, so every probe should be rejected.
    These are real people but only 2 'unknown' identities per fold; see report.
    """
    cols = embedding_columns(features)
    members = sorted(features["member_id"].unique())
    rows = []
    for u in members:
        others = [m for m in members if m != u]
        enrol = features[features["member_id"].isin(others)]
        if not use_aug_in_enrolment:
            enrol = enrol[enrol["is_original"]]
        ids, T = build_templates(enrol, cols, others)
        for r in features[(features["member_id"] == u) & features["is_original"]].itertuples(index=False):
            s = cosine_scores(np.array([getattr(r, c) for c in cols]), T)
            d = decide(s, ids, threshold)
            rows.append(dict(protocol="leave_one_member_out_unknown", probe_file=r.image_filename, true_id=UNKNOWN,
                             held_out_member=u, **{f"score_{m}": float(v) for m, v in zip(ids, s)}, **d))
    return pd.DataFrame(rows)


def augmented_probe_trials(features: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Robustness check ONLY: probe with augmented copies of the held-out original.

    These are not independent samples (same photo, transformed) and are reported
    separately from the headline metrics.
    """
    cols = embedding_columns(features)
    members = sorted(features["member_id"].unique())
    rows = []
    for e in sorted(features["expression"].unique()):
        enrol = features[features["expression"] != e]
        ids, T = build_templates(enrol, cols, members)
        for r in features[(features["expression"] == e) & ~features["is_original"]].itertuples(index=False):
            s = cosine_scores(np.array([getattr(r, c) for c in cols]), T)
            d = decide(s, ids, threshold)
            rows.append(dict(protocol="augmented_probe_robustness", probe_file=r.image_filename, true_id=r.member_id,
                             held_out_expression=e, augmentation_type=r.augmentation_type, **d))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- metrics
def verification_rates(loeo: pd.DataFrame, unknown: pd.DataFrame, threshold: float, members) -> dict:
    """Genuine / impostor accounting for verification.

    genuine trial   : probe vs the template of its OWN identity  -> should accept
    impostor trial  : probe vs the template of ANOTHER enrolled member -> should reject
    unknown trial   : probe from a never-enrolled person vs any enrolled template -> should reject
    """
    gen, imp = [], []
    for r in loeo.itertuples(index=False):
        for m in members:
            (gen if m == r.true_id else imp).append(getattr(r, f"score_{m}"))
    unk = [x for r in unknown.itertuples(index=False) for x in [r.score]]  # best score per unknown probe
    gen, imp, unk = np.array(gen), np.array(imp), np.array(unk)
    out = {"threshold": threshold, "n_genuine": int(len(gen)), "n_impostor": int(len(imp)), "n_unknown": int(len(unk))}
    if len(gen):
        k = int((gen < threshold).sum())
        out.update(FRR=k / len(gen), FRR_ci95=wilson_interval(k, len(gen)), genuine_accept=int((gen >= threshold).sum()))
    if len(imp):
        k = int((imp >= threshold).sum())
        out.update(FAR_enrolled_impostor=k / len(imp), FAR_enrolled_impostor_ci95=wilson_interval(k, len(imp)),
                   impostor_reject=int((imp < threshold).sum()))
    if len(unk):
        k = int((unk >= threshold).sum())
        out.update(FAR_unknown=k / len(unk), FAR_unknown_ci95=wilson_interval(k, len(unk)), unknown_reject=int((unk < threshold).sum()))
    out["min_genuine_score"] = float(gen.min()) if len(gen) else None
    out["max_impostor_score"] = float(imp.max()) if len(imp) else None
    out["max_unknown_best_score"] = float(unk.max()) if len(unk) else None
    out["scores_separable"] = (bool(gen.min() > max(imp.max() if len(imp) else -1, unk.max() if len(unk) else -1))
                               if len(gen) else None)
    return out


def open_set_classification(loeo: pd.DataFrame, unknown: pd.DataFrame, threshold: float, members) -> dict:
    """4-class (3 members + 'unknown') system-level metrics using final decisions."""
    t = pd.concat([loeo, unknown], ignore_index=True)
    y_true = t["true_id"].tolist()
    y_pred = [u if u is not None else UNKNOWN for u in t["user_id"]]
    labels = list(members) + [UNKNOWN]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return dict(n_samples=len(t), accuracy=float(accuracy_score(y_true, y_pred)), labels=labels,
                confusion_matrix=cm.tolist(),
                per_class=[dict(label=l, precision=float(a), recall=float(b), f1=float(c), support=int(d))
                           for l, a, b, c, d in zip(labels, p, r, f, s)],
                macro_precision=float(pm), macro_recall=float(rm), macro_f1=float(fm))


def closed_set_identification(loeo: pd.DataFrame) -> dict:
    """Ignoring the threshold: is the nearest template the right member? (3-class)"""
    acc = float((loeo["best_match"] == loeo["true_id"]).mean()) if len(loeo) else None
    return dict(n=len(loeo), top1_accuracy=acc)


def threshold_sweep(loeo, unknown, members, thresholds=None) -> pd.DataFrame:
    """EXPLORATORY ONLY. Reports FAR/FRR at many thresholds; not used to pick the threshold."""
    thresholds = np.round(np.arange(0.0, 1.0001, 0.02), 2) if thresholds is None else thresholds
    rows = []
    for th in thresholds:
        v = verification_rates(loeo, unknown, float(th), members)
        far = [v.get("FAR_enrolled_impostor"), v.get("FAR_unknown")]
        far_all = None
        if v["n_impostor"] + v["n_unknown"]:
            k = (v.get("FAR_enrolled_impostor", 0) * v["n_impostor"]) + (v.get("FAR_unknown", 0) * v["n_unknown"])
            far_all = k / (v["n_impostor"] + v["n_unknown"])
        rows.append(dict(threshold=float(th), FRR=v.get("FRR"), FAR=far_all))
    return pd.DataFrame(rows)


def run_full_evaluation(features: pd.DataFrame, threshold: float) -> dict:
    members = sorted(features["member_id"].unique())
    loeo = leave_one_expression_out(features, threshold)
    unk = unknown_person_trials(features, threshold)
    aug = augmented_probe_trials(features, threshold)
    return dict(
        loeo=loeo, unknown=unk, augmented=aug,
        closed_set=closed_set_identification(loeo),
        verification=verification_rates(loeo, unk, threshold, members),
        open_set=open_set_classification(loeo, unk, threshold, members),
        sweep=threshold_sweep(loeo, unk, members),
        augmented_robustness=dict(n=len(aug), accepted_as_correct_member=float(((aug["user_id"] == aug["true_id"])).mean()) if len(aug) else None),
    )
