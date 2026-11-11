"""
product recommendation component

What it does
------------
1. Loads the two raw datasets (customer_transactions, customer_social_profiles).
2. Applies the SAME cleaning and customer-level aggregation as the notebook
   (01_tabular_preprocessing.ipynb) so that features are built exactly as in training.
3. Loads the trained model bundle (models/product_model.joblib).
4. Builds the feature row of the requested customer and returns the predicted product.

Quick test from the repository root:
    python src/product_model.py --customer 100
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/formative2")     # used automatically on Colab

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


# --------------------------------------------------------------------------- cleaning (same as notebook)
def _clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    t = raw.copy().rename(columns={"customer_id_legacy": "customer_id"})
    for c in ["customer_id", "transaction_id", "purchase_amount", "customer_rating"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    if pd.api.types.is_numeric_dtype(t["purchase_date"]):                 # Excel serial numbers
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


# --------------------------------------------------------------------------- model training (same as notebook)
RANDOM_STATE = 42
MODEL_NUMERIC = ["avg_engagement", "avg_interest", "pct_Positive", "pct_Neutral", "pct_Negative",
                 "n_platforms", "has_social_profile", "engagement_interest",
                 "purchase_month", "purchase_weekday", "is_weekend",
                 "prev_purchases", "prev_avg_amount", "prev_avg_rating",
                 "prev_Books", "prev_Clothing", "prev_Electronics", "prev_Groceries", "prev_Sports"]
MODEL_CATEGORICAL = ["top_platform"]
MODEL_FEATURES = MODEL_NUMERIC + MODEL_CATEGORICAL          # the 20 features kept by the feature selection


def build_training_table(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Merged table with the leakage-safe engineered features (notebook Sections 5 and 7)."""
    trans, social_agg = _load_data(str(data_dir) if data_dir else None)
    merged = trans.merge(social_agg, on="customer_id", how="left", indicator=True, validate="many_to_one")
    merged["has_social_profile"] = (merged["_merge"] == "both").astype(int)
    merged = merged.drop(columns="_merge")
    merged["top_platform"] = merged["top_platform"].fillna("Unknown")
    assert len(merged) == len(trans), "merge changed the number of rows"

    df = merged.sort_values(["purchase_date", "transaction_id"]).reset_index(drop=True)
    df["purchase_month"] = df["purchase_date"].dt.month
    df["purchase_weekday"] = df["purchase_date"].dt.day_of_week
    df["is_weekend"] = (df["purchase_weekday"] >= 5).astype(int)
    df["engagement_interest"] = df["avg_engagement"] * df["avg_interest"] / 100

    g = df.groupby("customer_id")                       # history = earlier purchases only
    df["prev_purchases"] = g.cumcount()
    df["prev_avg_amount"] = g["purchase_amount"].transform(lambda x: x.shift().expanding().mean())
    df["prev_avg_rating"] = g["customer_rating"].transform(lambda x: x.shift().expanding().mean())

    dummies = pd.get_dummies(df["product_category"]).astype(int)
    prev_cat = dummies.groupby(df["customer_id"]).cumsum() - dummies
    prev_cat.columns = [f"prev_{c}" for c in prev_cat.columns]
    return pd.concat([df, prev_cat], axis=1)


def train_and_save(model_path: str | Path | None = None, data_dir: str | Path | None = None) -> dict:
    """Train the final model (Logistic Regression, C = 0.02) on all rows and save the bundle."""
    df = build_training_table(data_dir)
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    assert not missing, f"missing feature columns: {missing}"
    prev_cat_cols = [c for c in df.columns if c.startswith("prev_") and c.replace("prev_", "", 1)
                     in set(df["product_category"])]

    le = LabelEncoder()
    y = le.fit_transform(df["product_category"])
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), MODEL_NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), MODEL_CATEGORICAL),
    ])
    pipe = Pipeline([("prep", preprocessor),
                     ("model", LogisticRegression(max_iter=3000, C=0.02, random_state=RANDOM_STATE))])
    pipe.fit(df[MODEL_FEATURES], y)

    bundle = {"pipeline": pipe, "label_encoder": le, "feature_columns": MODEL_FEATURES,
              "numeric": MODEL_NUMERIC, "categorical": MODEL_CATEGORICAL, "model_name": "Logistic Regression",
              "classes": list(le.classes_), "prev_cat_cols": prev_cat_cols}
    target = Path(model_path) if model_path else MODEL_PATH_CANDIDATES[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target)
    print(f"[product_model] Model trained on {len(df)} rows (scikit-learn {sklearn.__version__}) "
          f"and saved to {target}", file=sys.stderr)
    return bundle


@lru_cache(maxsize=4)
def load_model(model_path: str | None = None) -> dict:
    """
    Load the saved bundle (pipeline, label encoder, feature list, class names).
    If the file is missing, corrupted, or was saved with another scikit-learn version,
    the model is rebuilt from the raw data and saved again.
    """
    try:
        found = _find_model(model_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)      # a version mismatch = rebuild
            return joblib.load(found)
    except Exception as exc:                                                 # missing / other version / corrupt
        print(f"[product_model] Saved model not usable ({type(exc).__name__}); "
              "rebuilding it from the raw data...", file=sys.stderr)
        return train_and_save(model_path)


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
    row["purchase_weekday"] = as_of.day_of_week
    row["is_weekend"] = int(as_of.day_of_week >= 5)
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
    parser.add_argument("--retrain", action="store_true", help="rebuild and save the model from the raw data")
    parser.add_argument("--data-dir", default=None, help="folder with the raw data files")
    parser.add_argument("--model", default=None, help="path to product_model.joblib")
    args = parser.parse_args()

    if args.retrain:
        train_and_save(args.model, args.data_dir)
        load_model.cache_clear()
    if args.list:
        ids = known_customer_ids(args.data_dir)
        print(f"{len(ids)} known customers: {ids}")
    if args.customer is not None:
        print(json.dumps(predict_product(args.customer, args.data_dir, args.model), indent=2))
    if not (args.list or args.retrain) and args.customer is None:
        parser.print_help()


if __name__ == "__main__":
    _main()
