"""Image ingestion, inspection and augmentation utilities (no deep-learning dependency).

Everything here works with Pillow / NumPy / OpenCV only, so it can be run and
tested without TensorFlow or DeepFace.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

MEMBERS = ["member_1", "member_2", "member_3"]
EXPRESSIONS = ["neutral", "smiling", "surprised"]
AUGMENTATIONS = ["rotation", "flip", "grayscale", "brightness"]
IMAGE_EXTS = {".jpg", ".jpeg", ".jepg", ".png", ".webp", ".bmp"}


# ----------------------------------------------------------------- helpers
def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_member_and_expression(rel_path: str):
    """Infer (member_id, expression) from a path such as 'Member 2/neutral.jepg'.

    Returns None for a field that cannot be determined (never guesses).
    """
    p = rel_path.lower().replace("\\", "/")
    m = re.search(r"member[\s_\-]*([0-9]+)", p)
    member = f"member_{int(m.group(1))}" if m else None
    expr = None
    for key, name in (("neutral", "neutral"), ("smil", "smiling"), ("surpris", "surprised")):
        if key in p:
            expr = name
            break
    return member, expr


def load_image_rgb(path, max_side: int | None = None) -> np.ndarray:
    """Load an image as an RGB uint8 array (correct EXIF orientation).

    Raises ValueError for unreadable files so callers can handle it gracefully.
    PIL decodes to RGB, so no accidental BGR display when used with matplotlib.
    """
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            if max_side and max(im.size) > max_side:
                scale = max_side / max(im.size)
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
            return np.asarray(im, dtype=np.uint8).copy()
    except Exception as exc:  # corrupted, wrong format, missing file ...
        raise ValueError(f"Cannot read image '{path}': {exc}") from exc


def save_rgb(arr: np.ndarray, path, quality: int = 95) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=quality)


# --------------------------------------------------------------- ingestion
def ingest_zip(zip_path, raw_dir, mapping_csv=None) -> pd.DataFrame:
    """Extract an uploaded dataset ZIP into the canonical layout.

    * member_id and expression are inferred from folder/file names.
    * files are COPIED byte-for-byte (originals are not re-encoded) to
      ``raw_dir/member_N/member_N_<expression>.jpg``; only the name changes.
    * every file in the ZIP is listed in the returned mapping; nothing is dropped
      silently (unrecognised files are flagged with status != 'ok').
    """
    zip_path, raw_dir = Path(zip_path), Path(raw_dir)
    tmp = raw_dir.parent / "_zip_extract_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    rows = []
    for f in sorted(p for p in tmp.rglob("*") if p.is_file() and "__MACOSX" not in p.parts):
        rel = str(f.relative_to(tmp))
        row = {"zip_relative_path": rel, "original_extension": f.suffix.lower()}
        if f.suffix.lower() not in IMAGE_EXTS:
            row.update(status="skipped_not_an_image", member_id=None, expression=None, canonical_path=None)
        else:
            member, expr = parse_member_and_expression(rel)
            if member is None or expr is None:
                row.update(status="unrecognised_member_or_expression", member_id=member, expression=expr, canonical_path=None)
            else:
                dest = raw_dir / member / f"{member}_{expr}.jpg"
                if dest.exists():
                    row.update(status="duplicate_target_name", member_id=member, expression=expr, canonical_path=None)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(f, dest)  # byte-identical copy
                    row.update(status="ok", member_id=member, expression=expr,
                               canonical_path=f"{member}/{dest.name}")
        rows.append(row)
    shutil.rmtree(tmp, ignore_errors=True)
    df = pd.DataFrame(rows)
    if mapping_csv:
        Path(mapping_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(mapping_csv, index=False)
    return df


# -------------------------------------------------------------- inspection
def inspect_dataset(raw_dir) -> pd.DataFrame:
    """One row per discovered image with readability, size, format and duplicate checks."""
    raw_dir = Path(raw_dir)
    rows = []
    for f in sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        member, expr = parse_member_and_expression(f.stem)
        member = member or (f.parent.name if f.parent.name.startswith("member_") else None)
        row = dict(member_id=member, expression=expr, image_filename=f.name, path=str(f),
                   file_size_kb=round(f.stat().st_size / 1024, 1), sha256=sha256_of(f))
        try:
            with Image.open(f) as im:
                row.update(format=im.format, width=im.width, height=im.height, mode=im.mode,
                           channels=len(im.getbands()), exif_orientation=im.getexif().get(274, 1))
                im.load()
            row.update(readable=True, error="")
        except Exception as exc:
            row.update(readable=False, error=str(exc))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    first = {}
    dup = []
    for h in df["sha256"]:
        dup.append(first.get(h))
        first.setdefault(h, df.loc[df["sha256"] == h, "image_filename"].iloc[0])
    df["duplicate_of"] = [d if d != n else None for d, n in zip(dup, df["image_filename"])]
    return df


def missing_items(df: pd.DataFrame, members=MEMBERS, expressions=EXPRESSIONS) -> list[tuple[str, str]]:
    have = set(zip(df["member_id"], df["expression"])) if len(df) else set()
    return [(m, e) for m in members for e in expressions if (m, e) not in have]


# ------------------------------------------------------------ augmentation
def augment_image(rgb: np.ndarray, kind: str) -> np.ndarray:
    """Identity-preserving augmentations with deliberately mild parameters.

    rotation   : +8 deg about the centre, reflected borders (no black corners)
    flip       : horizontal mirror
    grayscale  : luminance, replicated to 3 channels so shape matches the model input
    brightness : x1.20 gain, clipped to [0, 255]
    """
    import cv2

    if kind == "rotation":
        h, w = rgb.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.0)
        return cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if kind == "flip":
        return np.ascontiguousarray(rgb[:, ::-1])
    if kind == "grayscale":
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return np.stack([g, g, g], axis=-1)
    if kind == "brightness":
        return np.clip(rgb.astype(np.float32) * 1.20, 0, 255).astype(np.uint8)
    raise ValueError(f"Unknown augmentation '{kind}'")


def generate_augmentations(inspect_df: pd.DataFrame, out_dir, max_side: int = 1024,
                           kinds=AUGMENTATIONS) -> pd.DataFrame:
    """Write augmented copies (never touching originals) and return a manifest."""
    out_dir = Path(out_dir)
    rows = []
    for r in inspect_df[inspect_df["readable"]].itertuples():
        stem = Path(r.image_filename).stem
        rgb = load_image_rgb(r.path, max_side=max_side)
        for kind in kinds:
            name = f"{stem}_{kind}.jpg"
            dest = out_dir / r.member_id / name
            save_rgb(augment_image(rgb, kind), dest)
            rows.append(dict(member_id=r.member_id, expression=r.expression,
                             original_filename=r.image_filename, augmentation_type=kind,
                             output_filename=name, output_path=str(dest), source_max_side=max_side))
    return pd.DataFrame(rows)
