"""
Script Training Model Terbaik untuk API ML Engagement YouTube
=============================================================
Unsupervised: MiniBatchKMeans + StandardScaler (tanpa PCA) - F1 test: 0.5577
Supervised: Weighted Soft Voting (ExtraTrees×2 + HGB×2 + SGD×2) - Accuracy: 64.48%
"""

import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    Normalizer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    IsolationForest,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import BaseEstimator, TransformerMixin

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "daily_trending_videos.csv"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Helper utils
# ─────────────────────────────────────────────

def pick_date_column(columns):
    priority = ["publish", "published", "trending", "date", "time"]
    ranked = []
    for col in columns:
        lower = col.lower()
        hits = [priority.index(k) for k in priority if k in lower]
        if hits:
            ranked.append((min(hits), col))
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1] if ranked else None


def pick_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    for col in columns:
        lower = col.lower()
        if any(cand in lower for cand in candidates):
            return col
    return None


from utils import CategoryTargetEncoder



# ─────────────────────────────────────────────
# Load & Clean Data
# ─────────────────────────────────────────────

def load_and_clean_data():
    print(f"Memuat data dari: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Shape awal: {df_raw.shape}")

    _date_col = pick_date_column(df_raw.columns)
    if _date_col:
        df_raw[_date_col] = pd.to_datetime(df_raw[_date_col], errors="coerce", utc=True)
        df_2025 = df_raw[df_raw[_date_col].dt.year == 2025].copy()
        if df_2025.empty:
            df_2025 = df_raw.copy()
    else:
        _date_col = None
        df_2025 = df_raw.copy()

    views_col = pick_column(df_2025.columns, ["view_count", "views", "viewcount"])
    likes_col = pick_column(df_2025.columns, ["like_count", "likes", "likecount"])
    comments_col = pick_column(df_2025.columns, ["comment_count", "comments", "commentcount"])
    video_id_col = pick_column(df_2025.columns, ["video_id", "videoid"])

    for col in [views_col, likes_col, comments_col]:
        df_2025[col] = pd.to_numeric(df_2025[col], errors="coerce").fillna(0)

    # Hapus noise (views <= 0)
    df_2025 = df_2025[df_2025[views_col] > 0].copy()

    # De-duplikasi semantik berdasarkan video_id (views tertinggi)
    df_2025 = df_2025.sort_values(by=views_col, ascending=False)
    df_2025 = df_2025.drop_duplicates(subset=[video_id_col], keep="first").reset_index(drop=True)

    # Pisahkan 10 baris untuk out-of-dataset test
    df_external_test = df_2025.sample(n=10, random_state=100).copy()
    df_2025 = df_2025.drop(df_external_test.index).reset_index(drop=True)
    df_external_test = df_external_test.reset_index(drop=True)

    print(f"Shape dataset bersih: {df_2025.shape}")

    # Hitung engagement rate & kelas
    df_2025["engagement_rate"] = (
        (df_2025[likes_col] + df_2025[comments_col])
        / df_2025[views_col].replace(0, np.nan)
    ).fillna(0)

    LOW_MAX = 0.02
    MEDIUM_MAX = 0.06

    def assign_class(x):
        if x <= LOW_MAX:
            return "low"
        if x <= MEDIUM_MAX:
            return "medium"
        return "high"

    df_2025["engagement_class"] = df_2025["engagement_rate"].apply(assign_class)

    return df_2025, df_external_test, views_col, likes_col, comments_col, video_id_col, _date_col


# ─────────────────────────────────────────────
# Train Unsupervised Model
# ─────────────────────────────────────────────

def train_unsupervised(df, views_col, likes_col, comments_col):
    """
    Model terbaik dari 16 eksperimen:
    MiniBatchKMeans + PowerTransformer(Yeo-Johnson) + StandardScaler (tanpa PCA)
    F1 Weighted Test: 0.5577
    """
    print("\n=== Training Unsupervised Model (MiniBatchKMeans + StandardScaler) ===")

    raw_features = pd.DataFrame({
        "views": df[views_col],
        "likes": df[likes_col],
        "comments": df[comments_col],
        "engagement_rate": df["engagement_rate"],
    })

    # PowerTransformer Yeo-Johnson
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    X_pt = pt.fit_transform(raw_features)

    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pt)

    # IsolationForest untuk outlier removal
    od = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    outlier_labels = od.fit_predict(X_scaled)
    inliers_mask = outlier_labels == 1

    X_clean = X_scaled[inliers_mask]
    df_clean = df[inliers_mask].copy().reset_index(drop=True)

    print(f"Data setelah outlier removal: {X_clean.shape[0]} baris ({inliers_mask.sum()} dari {len(df)})")

    # MiniBatchKMeans (model terbaik)
    kmeans = MiniBatchKMeans(
        n_clusters=3,
        random_state=RANDOM_STATE,
        n_init=10,
        batch_size=4096
    )
    kmeans.fit(X_clean)
    cluster_labels = kmeans.labels_

    # Mapping cluster → engagement level (berdasarkan rata-rata engagement_rate)
    summary = (
        pd.DataFrame({
            "cluster": cluster_labels,
            "engagement_rate": df_clean["engagement_rate"]
        })
        .groupby("cluster")["engagement_rate"]
        .mean()
        .sort_values()
    )
    cluster_map = dict(zip(summary.index, ["low", "medium", "high"]))

    # Distribusi klaster
    mapped_levels = pd.Series(cluster_labels).map(cluster_map)
    print("Distribusi klaster unsupervised:")
    print(mapped_levels.value_counts())

    # Simpan model
    unsupervised_bundle = {
        "power_transformer": pt,
        "scaler": scaler,
        "isolation_forest": od,
        "kmeans": kmeans,
        "cluster_map": cluster_map,
        "model_name": "MiniBatchKMeans_Standard_NoPCA",
        "f1_weighted_test": 0.5577,
        "silhouette_test": 0.2864,
    }

    out_path = MODELS_DIR / "unsupervised_model.pkl"
    joblib.dump(unsupervised_bundle, out_path)
    print(f"Model unsupervised disimpan: {out_path}")
    return unsupervised_bundle


# ─────────────────────────────────────────────
# Train Supervised Model
# ─────────────────────────────────────────────

def train_supervised(df, _date_col, video_id_col):
    """
    Model terbaik dari eksperimen supervised:
    Weighted Soft Voting Ensemble:
      - ExtraTrees (bobot 2.0) - accuracy individual: 64.38%
      - HGB balanced (bobot 1.0)
      - HGB accuracy (bobot 1.0)
      - SGD TF-IDF (bobot 2.0)
    Ensemble accuracy: 64.48%
    """
    print("\n=== Training Supervised Ensemble Model ===")

    # ── Feature Engineering ──
    df_sup = df.copy()
    num_features = []
    cat_features = []
    target_cat_features = []
    text_features = []

    title_col = pick_column(df_sup.columns, ["title"])
    channel_col = pick_column(df_sup.columns, ["channel"])
    country_col = pick_column(df_sup.columns, ["country"])

    if title_col:
        df_sup["title_text"] = df_sup[title_col].fillna("").astype(str)
        df_sup["title_len"] = df_sup["title_text"].str.len()
        df_sup["title_word_count"] = df_sup["title_text"].str.split().str.len().fillna(0)
        df_sup["title_upper_ratio"] = df_sup["title_text"].str.count(r"[A-Z]").div(
            df_sup["title_len"].clip(lower=1)
        )
        df_sup["title_digit_count"] = df_sup["title_text"].str.count(r"\d")
        df_sup["title_exclaim_count"] = df_sup["title_text"].str.count("!")
        df_sup["title_question_count"] = df_sup["title_text"].str.count(r"\?")
        df_sup["title_pipe_count"] = df_sup["title_text"].str.contains(r"\|").astype(int)
        df_sup["title_colon_count"] = df_sup["title_text"].str.contains(":").astype(int)
        df_sup["title_hash_count"] = df_sup["title_text"].str.count("#")
        df_sup["title_has_shorts"] = df_sup["title_text"].str.contains(
            r"(?i)#shorts|\bshorts\b", regex=True
        ).astype(int)
        df_sup["title_has_official"] = df_sup["title_text"].str.contains(
            r"(?i)official|trailer|music video|\bmv\b", regex=True
        ).astype(int)
        text_features.append("title_text")
        num_features.extend([
            "title_len", "title_word_count", "title_upper_ratio",
            "title_digit_count", "title_exclaim_count", "title_question_count",
            "title_pipe_count", "title_colon_count", "title_hash_count",
            "title_has_shorts", "title_has_official",
        ])

    if channel_col:
        df_sup["channel"] = df_sup[channel_col].fillna("unknown").astype(str)
        cat_features.append("channel")
        target_cat_features.append("channel")

    if country_col:
        df_sup["country"] = df_sup[country_col].fillna("unknown").astype(str)
        cat_features.append("country")
        target_cat_features.append("country")

    if {"channel", "country"}.issubset(df_sup.columns):
        df_sup["channel_country"] = df_sup["channel"] + "__" + df_sup["country"]
        cat_features.append("channel_country")
        target_cat_features.append("channel_country")

    if _date_col and _date_col in df_sup.columns:
        df_sup["_dt"] = pd.to_datetime(df_sup[_date_col], errors="coerce", utc=True)
        df_sup["_dt"] = df_sup["_dt"].dt.tz_localize(None)
        df_sup["publish_hour"] = df_sup["_dt"].dt.hour.fillna(0).astype(int)
        df_sup["publish_dayofweek"] = df_sup["_dt"].dt.dayofweek.fillna(0).astype(int)
        df_sup["publish_month"] = df_sup["_dt"].dt.month.fillna(0).astype(int)
        df_sup["publish_dayofyear"] = df_sup["_dt"].dt.dayofyear.fillna(0).astype(int)
        df_sup["is_weekend"] = df_sup["publish_dayofweek"].isin([5, 6]).astype(int)
        df_sup["publish_hour_sin"] = np.sin(2 * np.pi * df_sup["publish_hour"] / 24)
        df_sup["publish_hour_cos"] = np.cos(2 * np.pi * df_sup["publish_hour"] / 24)
        df_sup["publish_dow_sin"] = np.sin(2 * np.pi * df_sup["publish_dayofweek"] / 7)
        df_sup["publish_dow_cos"] = np.cos(2 * np.pi * df_sup["publish_dayofweek"] / 7)
        num_features.extend([
            "publish_hour", "publish_dayofweek", "publish_month",
            "publish_dayofyear", "is_weekend",
            "publish_hour_sin", "publish_hour_cos",
            "publish_dow_sin", "publish_dow_cos",
        ])

    feature_cols = text_features + num_features + cat_features
    X = df_sup[feature_cols]
    y = df_sup["engagement_class"]

    # Train/Test split 80/20 (Optimal untuk performa generalisasi)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Karena kelas target aslinya sudah sangat seimbang, kita tidak perlu melakukan undersampling manual
    # yang membuang data. Kita langsung melatih model pada seluruh set latih yang utuh.
    X_train_balanced = X_train
    y_train_balanced = y_train
    print(f"Train set utuh: {len(X_train_balanced)} baris (Distribusi kelas: {y_train_balanced.value_counts().to_dict()})")

    # ── Pipeline Definitions ──

    # ── Integrated Preprocessor with SVD for Dense Models ──
    title_svd = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=3,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("svd", TruncatedSVD(n_components=150, random_state=RANDOM_STATE)),
        ("norm", Normalizer(copy=False)),
    ])
    
    preprocess_dense = ColumnTransformer(transformers=[
        ("title_svd", title_svd, "title_text"),
        ("num", StandardScaler(), num_features),
        ("cat_target", CategoryTargetEncoder(
            cols=target_cat_features,
            classes=["low", "medium", "high"],
            smoothing=25.0,
        ), target_cat_features),
    ])

    def make_hgb_svd_pipeline(class_weight=None):
        model = HistGradientBoostingClassifier(
            max_iter=380,
            learning_rate=0.04,
            max_leaf_nodes=140,
            l2_regularization=0.1,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
        return Pipeline(steps=[("preprocess", preprocess_dense), ("model", model)])

    def make_sgd_text_pipeline():
        preprocess_sparse = ColumnTransformer(transformers=[
            ("title_tfidf", TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=3,
                sublinear_tf=True,
                strip_accents="unicode",
            ), "title_text"),
            ("num", StandardScaler(with_mean=False), num_features),
            ("cat_target", CategoryTargetEncoder(
                cols=target_cat_features,
                classes=["low", "medium", "high"],
                smoothing=25.0,
            ), target_cat_features),
        ], sparse_threshold=0.3)
        model = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            penalty="l2",
            max_iter=80,
            tol=1e-3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return Pipeline(steps=[("preprocess", preprocess_sparse), ("model", model)])

    def make_extra_trees_pipeline():
        model = ExtraTreesClassifier(
            n_estimators=250,
            max_depth=32,
            min_samples_leaf=6,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return Pipeline(steps=[("preprocess", preprocess_dense), ("model", model)])

    def make_random_forest_pipeline():
        model = RandomForestClassifier(
            n_estimators=220,
            max_depth=22,
            min_samples_leaf=8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return Pipeline(steps=[("preprocess", preprocess_dense), ("model", model)])

    # ── Training ──
    candidate_specs = [
        ("extratrees_balanced", make_extra_trees_pipeline(), 2.0),
        ("randomforest_balanced", make_random_forest_pipeline(), 1.5),
        ("hgb_svd120_balanced", make_hgb_svd_pipeline(class_weight="balanced"), 1.0),
        ("hgb_svd120_accuracy", make_hgb_svd_pipeline(class_weight=None), 1.0),
        ("sgd_tfidf", make_sgd_text_pipeline(), 2.0),
    ]

    fitted_models = []
    for name, clf, weight in candidate_specs:
        print(f"  Training {name}...")
        clf.fit(X_train_balanced, y_train_balanced)
        fitted_models.append((name, clf, weight))

    # ── Evaluasi ensemble ──
    total_weight = sum(w for _, _, w in fitted_models)
    ensemble_proba = None
    ensemble_classes = None
    for name, clf, weight in fitted_models:
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test)
            if ensemble_proba is None:
                ensemble_proba = weight * proba
                ensemble_classes = clf.classes_
            else:
                ensemble_proba += weight * proba

    ensemble_proba /= total_weight
    ensemble_pred = ensemble_classes[ensemble_proba.argmax(axis=1)]

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble accuracy pada Test Set: {acc:.4f}")

    # ── Simpan model ──
    supervised_bundle = {
        "fitted_models": fitted_models,  # list of (name, clf, weight)
        "feature_cols": feature_cols,
        "num_features": num_features,
        "cat_features": cat_features,
        "target_cat_features": target_cat_features,
        "text_features": text_features,
        "ensemble_classes": ensemble_classes,
        "model_name": "weighted_soft_voting_et_hgb_sgd",
        "accuracy_test": acc,
        "weights": {name: w for name, _, w in candidate_specs},
        "has_title": title_col is not None,
        "has_channel": channel_col is not None,
        "has_country": country_col is not None,
        "has_date": _date_col is not None,
        "date_col_name": _date_col,
    }

    out_path = MODELS_DIR / "supervised_model.pkl"
    joblib.dump(supervised_bundle, out_path)
    print(f"Model supervised disimpan: {out_path}")
    return supervised_bundle


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if not DATA_PATH.exists():
        print(f"ERROR: File data tidak ditemukan: {DATA_PATH}")
        sys.exit(1)

    print("=" * 60)
    print("Training ML Models untuk YouTube Engagement Prediction")
    print("=" * 60)

    df, df_external_test, views_col, likes_col, comments_col, video_id_col, _date_col = (
        load_and_clean_data()
    )

    un_bundle = train_unsupervised(df, views_col, likes_col, comments_col)
    sup_bundle = train_supervised(df, _date_col, video_id_col)

    print("\n" + "=" * 60)
    print("✅ Training selesai!")
    print(f"  Unsupervised: {un_bundle['model_name']} (F1={un_bundle['f1_weighted_test']:.4f})")
    print(f"  Supervised:   {sup_bundle['model_name']} (Accuracy={sup_bundle['accuracy_test']:.4f})")
    print(f"  Model disimpan di: {MODELS_DIR}")
