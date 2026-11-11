"""
product recommendation component.

What it does
------------
1. Loads the two raw datasets (customer_transactions, customer_social_profiles).
2. Applies the SAME cleaning and customer-level aggregation as the notebook
   (01_tabular_preprocessing.ipynb) so that features are built exactly as in training.
3. Loads the trained model bundle (models/product_model.joblib).
4. Builds the feature row of the requested customer and returns the predicted product.

"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/formative2")     

DATA_DIR_CANDIDATES = [REPO_ROOT / "data" / "raw", Path("data/raw"),
                       DRIVE_PROJECT_DIR / "data" / "raw"]
MODEL_PATH_CANDIDATES = [REPO_ROOT / "models" / "product_model.joblib", Path("models/product_model.joblib"),
                         DRIVE_PROJECT_DIR / "models" / "product_model.joblib"]

SOCIAL_NUMERIC_COLS = ["avg_engagement", "avg_interest", "pct_Positive", "pct_Neutral",
                       "pct_Negative", "n_platforms", "n_social_records"]
BRAND_CASING = {"linkedin": "LinkedIn", "tiktok": "TikTok"}


# --------------------------------------------------------------------------- file helpers
def _find_file(stem: str, data_dir: Path | None = None) -> Path:
    dirs = [Path(data_dir)] if data_dir else DATA_DIR_CANDIDATES
    for d in dirs:
        for ext in (".csv", ".xlsx"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"Could not find {stem}.csv or {stem}.xlsx in: {[str(d) for d in dirs]}")


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)


def _find_model(model_path: str | Path | None = None) -> Path:
    candidates = [Path(model_path)] if model_path else MODEL_PATH_CANDIDATES
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Model file not found. Looked in: {[str(p) for p in candidates]}. "
                            "Run the notebook (Section 11.2) to create models/product_model.joblib.")


# --------------------------------------------------------------------------- cleaning 
def _clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    t = raw.copy().rename(columns={"customer_id_legacy": "customer_id"})
    for c in ["customer_id", "transaction_id", "purchase_amount", "customer_rating"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    if pd.api.types.is_numeric_dtype(t["purchase_date"]):                 
        t["purchase_date"] = pd.to_datetime(t["purchase_date"], unit="D", origin="1899-12-30")
    else:
        t["purchase_date"] = pd.to_datetime(t["purchase_date"], errors="coerce")
    t["product_category"] = t["product_category"].astype(str).str.strip().str.title()

    t = t.dropna(subset=["customer_id", "transaction_id", "purchase_date", "product_category"])
    bad_rating = t["customer_rating"].notna() & ~t["customer_rating"].between(1, 5)
    t.loc[bad_rating, "customer_rating"] = np.nan
    t = t[~(t["purchase_amount"].isna() | (t["purchase_amount"] <= 0))]
    t = t.drop_duplicates().drop_duplicates(subset="transaction_id")
    t[["customer_id", "transaction_id"]] = t[["customer_id", "transaction_id"]].astype(int)
    t["purchase_amount"] = t["purchase_amount"].astype(float)
    return t.reset_index(drop=True)


def _clean_social(raw: pd.DataFrame) -> pd.DataFrame:
    s = raw.copy()
    s["customer_id"] = pd.to_numeric(s["customer_id_new"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    for c in ["engagement_score", "purchase_interest_score"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=["customer_id"])
    s["social_media_platform"] = (s["social_media_platform"].astype(str).str.strip()
                                  .map(lambda x: BRAND_CASING.get(x.lower(), x.title())))
    s["review_sentiment"] = s["review_sentiment"].astype(str).str.strip().str.title()
    bad_eng = s["engagement_score"].notna() & ~s["engagement_score"].between(0, 100)
    bad_int = s["purchase_interest_score"].notna() & ~s["purchase_interest_score"].between(1, 5)
    s.loc[bad_eng, "engagement_score"] = np.nan
    s.loc[bad_int, "purchase_interest_score"] = np.nan
    s = s.drop_duplicates()
    s["customer_id"] = s["customer_id"].astype(int)
    return s.reset_index(drop=True)


def _mode_or_nan(series: pd.Series):
    m = series.mode()
    return m.iloc[0] if len(m) else np.nan


def _aggregate_social(social: pd.DataFrame) -> pd.DataFrame:
    """One row per customer (same aggregation as notebook Section 5)."""
    agg = social.groupby("customer_id").agg(
        avg_engagement=("engagement_score", "mean"),
        avg_interest=("purchase_interest_score", "mean"),
        top_platform=("social_media_platform", _mode_or_nan),
        n_platforms=("social_media_platform", "nunique"),
        n_social_records=("customer_id", "size"),
    ).reset_index()
    sent = (pd.crosstab(social["customer_id"], social["review_sentiment"], normalize="index")
              .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
              .add_prefix("pct_").reset_index())
    return agg.merge(sent, on="customer_id", how="left")


# --------------------------------------------------------------------------- cached loaders
@lru_cache(maxsize=4)
def _load_data(data_dir: str | None = None):
    trans = _clean_transactions(_read_table(_find_file("customer_transactions", data_dir)))
    social = _clean_social(_read_table(_find_file("customer_social_profiles", data_dir)))
    return trans, _aggregate_social(social)


@lru_cache(maxsize=4)
def load_model(model_path: str | None = None) -> dict:
    """Load the saved bundle (pipeline, label encoder, feature list, class names)."""
    return joblib.load(_find_model(model_path))


# --------------------------------------------------------------------------- public API
def known_customer_ids(data_dir: str | Path | None = None) -> list[int]:
    """All customer IDs that exist in the transactions or social profiles."""
    trans, social_agg = _load_data(str(data_dir) if data_dir else None)
    return sorted(set(trans["customer_id"]) | set(social_agg["customer_id"]))


def build_features(customer_id: int, data_dir: str | Path | None = None,
                   model_path: str | Path | None = None, as_of=None) -> pd.DataFrame:
    """Build the one-row feature table for a customer, in the exact column order used for training."""
    bundle = load_model(str(model_path) if model_path else None)
    trans, social_agg = _load_data(str(data_dir) if data_dir else None)
    customer_id = int(customer_id)

    cust_tx = trans[trans["customer_id"] == customer_id].sort_values("purchase_date")
    soc = social_agg[social_agg["customer_id"] == customer_id]
    if cust_tx.empty and soc.empty:
        raise ValueError(f"Customer {customer_id} was not found in the transactions or social profiles.")

    as_of = pd.Timestamp(as_of) if as_of is not None else trans["purchase_date"].max() + pd.Timedelta(days=1)
    row = {c: np.nan for c in bundle["feature_columns"]}

    if len(soc):
        s = soc.iloc[0]
        for c in SOCIAL_NUMERIC_COLS:
            row[c] = s[c]
        row["top_platform"] = s["top_platform"]
        row["has_social_profile"] = 1
        row["engagement_interest"] = s["avg_engagement"] * s["avg_interest"] / 100
    else:
        row["top_platform"] = "Unknown"
        row["has_social_profile"] = 0

    row["purchase_month"] = as_of.month
    row["purchase_weekday"] = as_of.dayofweek
    row["is_weekend"] = int(as_of.dayofweek >= 5)
    row["prev_purchases"] = len(cust_tx)
    row["prev_avg_amount"] = cust_tx["purchase_amount"].mean() if len(cust_tx) else np.nan
    row["prev_avg_rating"] = cust_tx["customer_rating"].mean() if len(cust_tx) else np.nan
    for c in bundle["prev_cat_cols"]:
        row[c] = int((cust_tx["product_category"] == c.replace("prev_", "", 1)).sum())
    # only used if the model was trained with the current-purchase features
    if "purchase_amount" in row:
        row["purchase_amount"] = row["prev_avg_amount"]
    if "customer_rating" in row:
        row["customer_rating"] = row["prev_avg_rating"]

    return pd.DataFrame([row])[bundle["feature_columns"]]


def predict_product(customer_id: int, data_dir: str | Path | None = None,
                    model_path: str | Path | None = None, as_of=None) -> dict:
    """
    Predict the product a verified customer is most likely to buy.

    Returns
    -------
    {"customer_id": int,
     "predicted_product": str,          # e.g. "Electronics"
     "confidence": float,               # probability of the predicted product
     "probabilities": {product: probability, ...}}   # sorted from most to least likely

    Raises ValueError if the customer is unknown.
    Note: the model is only slightly better than chance (about 0.2 for 5 classes),
    so the confidence is low and the result should be shown as a recommendation.
    """
    bundle = load_model(str(model_path) if model_path else None)
    X_new = build_features(customer_id, data_dir, model_path, as_of)
    proba = bundle["pipeline"].predict_proba(X_new)[0]
    order = np.argsort(proba)[::-1]
    return {"customer_id": int(customer_id),
            "predicted_product": bundle["classes"][order[0]],
            "confidence": round(float(proba[order[0]]), 3),
            "probabilities": {bundle["classes"][i]: round(float(proba[i]), 3) for i in order}}


# --------------------------------------------------------------------------- command line test
def _main() -> None:
    parser = argparse.ArgumentParser(description="Predict the product a customer is likely to buy.")
    parser.add_argument("--customer", type=int, help="customer ID to predict for")
    parser.add_argument("--list", action="store_true", help="list the known customer IDs")
    parser.add_argument("--data-dir", default=None, help="folder with the raw data files")
    parser.add_argument("--model", default=None, help="path to product_model.joblib")
    args = parser.parse_args()

    if args.list:
        ids = known_customer_ids(args.data_dir)
        print(f"{len(ids)} known customers: {ids}")
    if args.customer is not None:
        print(json.dumps(predict_product(args.customer, args.data_dir, args.model), indent=2))
    if not args.list and args.customer is None:
        parser.print_help()


if __name__ == "__main__":
    _main()