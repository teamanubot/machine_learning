"""
FastAPI ML Engagement YouTube - API Endpoint
============================================
Endpoint untuk pengujian kombinasi supervised dan unsupervised learning
Accessible at: api.rmun.tech/machine-learning
"""

import os
import re
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import CategoryTargetEncoder, extract_video_id, fetch_youtube_metadata



# ─────────────────────────────────────────────
# App & Config (Load .env manually if exists)
# ─────────────────────────────────────────────
try:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()
except Exception:
    pass



app = FastAPI(
    title="YouTube Engagement ML API",
    description="API untuk pengujian kombinasi Supervised & Unsupervised Learning klasifikasi engagement video YouTube",
    version="1.0.0",
    docs_url="/machine-learning/docs",
    redoc_url="/machine-learning/redoc",
    openapi_url="/machine-learning/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path(__file__).parent / "models"
STATIC_DIR = Path(__file__).parent / "static"

# ─────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────

_un_model = None
_sup_model = None


def get_unsupervised_model():
    global _un_model
    if _un_model is None:
        pkl_path = MODELS_DIR / "unsupervised_model.pkl"
        if not pkl_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Model unsupervised belum ditraining. Jalankan train_models.py terlebih dahulu."
            )
        _un_model = joblib.load(pkl_path)
    return _un_model


def get_supervised_model():
    global _sup_model
    if _sup_model is None:
        pkl_path = MODELS_DIR / "supervised_model.pkl"
        if not pkl_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Model supervised belum ditraining. Jalankan train_models.py terlebih dahulu."
            )
        _sup_model = joblib.load(pkl_path)
    return _sup_model


# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class UnsupervisedInput(BaseModel):
    views: int
    likes: int
    comments: int

    class Config:
        json_schema_extra = {
            "example": {
                "views": 1500000,
                "likes": 85000,
                "comments": 3200,
            }
        }


class SupervisedInput(BaseModel):
    title: str
    channel: Optional[str] = "unknown"
    country: Optional[str] = "unknown"
    published_at: Optional[str] = None  # ISO format: "2025-03-15T14:30:00"

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Gol Spektakuler Messi di Final Champions League! #shorts",
                "channel": "ESPN Football",
                "country": "US",
                "published_at": "2025-03-15T19:00:00",
            }
        }


class CombinedInput(BaseModel):
    # Supervised (metadata pra-tayang)
    title: str
    channel: Optional[str] = "unknown"
    country: Optional[str] = "unknown"
    published_at: Optional[str] = None

    # Unsupervised (metrik pasca-tayang)
    views: int
    likes: int
    comments: int

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Gol Spektakuler Messi di Final Champions League! #shorts",
                "channel": "ESPN Football",
                "country": "US",
                "published_at": "2025-03-15T19:00:00",
                "views": 1500000,
                "likes": 85000,
                "comments": 3200,
            }
        }


class YouTubeUrlInput(BaseModel):
    url: str

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            }
        }



# ─────────────────────────────────────────────
# Helper: Feature Engineering untuk Supervised
# ─────────────────────────────────────────────

def build_supervised_features(title: str, channel: str, country: str, published_at: Optional[str], bundle: dict):
    num_features = bundle["num_features"]
    target_cat_features = bundle["target_cat_features"]
    text_features = bundle["text_features"]

    row = {}

    # Fitur judul
    if "title_text" in text_features:
        t = str(title)
        row["title_text"] = t
        row["title_len"] = len(t)
        row["title_word_count"] = len(t.split())
        row["title_upper_ratio"] = len(re.findall(r"[A-Z]", t)) / max(len(t), 1)
        row["title_digit_count"] = len(re.findall(r"\d", t))
        row["title_exclaim_count"] = t.count("!")
        row["title_question_count"] = t.count("?")
        row["title_pipe_count"] = 1 if "|" in t else 0
        row["title_colon_count"] = 1 if ":" in t else 0
        row["title_hash_count"] = t.count("#")
        row["title_has_shorts"] = 1 if re.search(r"(?i)#shorts|\bshorts\b", t) else 0
        row["title_has_official"] = 1 if re.search(r"(?i)official|trailer|music video|\bmv\b", t) else 0

    # Fitur kategori
    if "channel" in target_cat_features:
        row["channel"] = str(channel) if channel else "unknown"
    if "country" in target_cat_features:
        row["country"] = str(country) if country else "unknown"
    if "channel_country" in target_cat_features:
        row["channel_country"] = f"{row.get('channel', 'unknown')}__{row.get('country', 'unknown')}"

    # Fitur tanggal
    if "publish_hour" in num_features:
        if published_at:
            try:
                dt = pd.to_datetime(published_at, utc=True).tz_localize(None)
            except Exception:
                try:
                    dt = pd.to_datetime(published_at).replace(tzinfo=None)
                except Exception:
                    dt = datetime.now()
        else:
            dt = datetime.now()

        row["publish_hour"] = dt.hour
        row["publish_dayofweek"] = dt.dayofweek
        row["publish_month"] = dt.month
        row["publish_dayofyear"] = dt.timetuple().tm_yday
        row["is_weekend"] = 1 if dt.dayofweek in [5, 6] else 0
        row["publish_hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
        row["publish_hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)
        row["publish_dow_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
        row["publish_dow_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)

    feature_cols = bundle["feature_cols"]
    X = pd.DataFrame([row])[feature_cols]
    return X


# ─────────────────────────────────────────────
# Helper: Predict Unsupervised
# ─────────────────────────────────────────────

def predict_cluster(views: int, likes: int, comments: int, bundle: dict) -> dict:
    engagement_rate = (likes + comments) / max(views, 1)

    raw = np.array([[views, likes, comments, engagement_rate]])

    pt = bundle["power_transformer"]
    scaler = bundle["scaler"]
    kmeans = bundle["kmeans"]
    cluster_map = bundle["cluster_map"]

    X_pt = pt.transform(raw)
    X_scaled = scaler.transform(X_pt)
    cluster_id = int(kmeans.predict(X_scaled)[0])
    level = cluster_map.get(cluster_id, "unknown")

    # Hitung jarak ke seluruh centroid klaster
    centers = kmeans.cluster_centers_
    distances = np.linalg.norm(X_scaled - centers, axis=1)
    
    # Hitung keanggotaan relatif menggunakan invers jarak (Soft Clustering Probability)
    # Ini memastikan confidence score menggambarkan keyakinan relatif dibanding klaster lain
    eps = 1e-6
    inv_distances = 1.0 / (distances + eps)
    relative_probs = inv_distances / np.sum(inv_distances)
    confidence_score = float(relative_probs[cluster_id])

    return {
        "cluster_id": cluster_id,
        "engagement_level": level,
        "engagement_rate": round(engagement_rate, 6),
        "confidence_score": round(confidence_score, 4),
        "cluster_map": cluster_map,
    }


# ─────────────────────────────────────────────
# Helper: Predict Supervised
# ─────────────────────────────────────────────

def predict_class(title: str, channel: str, country: str, published_at: Optional[str], bundle: dict) -> dict:
    X = build_supervised_features(title, channel, country, published_at, bundle)

    fitted_models = bundle["fitted_models"]
    ensemble_classes = bundle["ensemble_classes"]
    total_weight = sum(w for _, _, w in fitted_models)

    all_proba = None
    proba_per_model = {}

    for name, clf, weight in fitted_models:
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(X)[0]
            proba_per_model[name] = {
                cls: round(float(prob), 4)
                for cls, prob in zip(ensemble_classes, p)
            }
            weighted_p = weight * p
            if all_proba is None:
                all_proba = weighted_p
            else:
                all_proba += weighted_p

    all_proba /= total_weight
    pred_idx = int(np.argmax(all_proba))
    predicted_class = str(ensemble_classes[pred_idx])

    class_proba = {
        cls: round(float(prob), 4)
        for cls, prob in zip(ensemble_classes, all_proba)
    }
    confidence = round(float(all_proba[pred_idx]), 4)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_probabilities": class_proba,
        "model_probabilities": proba_per_model,
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/machine-learning/", response_class=HTMLResponse, tags=["UI"])
@app.get("/machine-learning", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve halaman UI pengujian model ML."""
    html_file = STATIC_DIR / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.get("/machine-learning/health", tags=["Info"])
async def health_check():
    """Cek status API dan ketersediaan model."""
    un_ok = (MODELS_DIR / "unsupervised_model.pkl").exists()
    sup_ok = (MODELS_DIR / "supervised_model.pkl").exists()

    result = {
        "status": "ok",
        "unsupervised_model_ready": un_ok,
        "supervised_model_ready": sup_ok,
    }

    if un_ok:
        try:
            bundle = get_unsupervised_model()
            result["unsupervised_info"] = {
                "model_name": bundle.get("model_name"),
                "f1_weighted_test": bundle.get("f1_weighted_test"),
                "silhouette_test": bundle.get("silhouette_test"),
            }
        except Exception:
            pass

    if sup_ok:
        try:
            bundle = get_supervised_model()
            result["supervised_info"] = {
                "model_name": bundle.get("model_name"),
                "accuracy_test": bundle.get("accuracy_test"),
                "weights": bundle.get("weights"),
            }
        except Exception:
            pass

    return result


@app.post("/machine-learning/predict/unsupervised", tags=["Prediksi"])
async def predict_unsupervised(data: UnsupervisedInput):
    """
    **Prediksi Unsupervised (Pasca-Tayang)**

    Mengelompokkan video berdasarkan metrik performa aktual (views, likes, comments)
    menggunakan model terbaik: **MiniBatchKMeans + StandardScaler** (tanpa PCA).

    - F1 Weighted Test: **0.5577**
    - Silhouette Test: **0.2864**
    """
    bundle = get_unsupervised_model()

    try:
        result = predict_cluster(data.views, data.likes, data.comments, bundle)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediksi gagal: {str(e)}")

    return {
        "model_type": "unsupervised",
        "model_name": bundle.get("model_name", "MiniBatchKMeans_Standard_NoPCA"),
        "input": {
            "views": data.views,
            "likes": data.likes,
            "comments": data.comments,
        },
        "output": result,
        "interpretation": {
            "low": "Engagement rendah: views banyak tapi interaksi minim",
            "medium": "Engagement sedang: keseimbangan antara views dan interaksi",
            "high": "Engagement tinggi: proporsi interaksi aktif sangat besar",
        }
    }


@app.post("/machine-learning/predict/supervised", tags=["Prediksi"])
async def predict_supervised(data: SupervisedInput):
    """
    **Prediksi Supervised (Pra-Tayang)**

    Memprediksi potensi kelas engagement berdasarkan metadata video sebelum dipublikasikan
    menggunakan model terbaik: **Weighted Soft Voting Ensemble**
    (ExtraTrees×2 + HGB×2 + SGD×2).

    - Accuracy Test: **64.48%**
    - Fitur: title, channel, country, published_at
    """
    bundle = get_supervised_model()

    try:
        result = predict_class(
            data.title, data.channel or "unknown",
            data.country or "unknown", data.published_at, bundle
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediksi gagal: {str(e)}")

    return {
        "model_type": "supervised",
        "model_name": bundle.get("model_name", "weighted_soft_voting_et_hgb_sgd"),
        "input": {
            "title": data.title,
            "channel": data.channel,
            "country": data.country,
            "published_at": data.published_at,
        },
        "output": result,
        "note": "Prediksi ini berdasarkan metadata awal tanpa melihat hasil tayang (bebas data leakage)",
    }


@app.post("/machine-learning/predict/combined", tags=["Prediksi"])
async def predict_combined(data: CombinedInput):
    """
    **Prediksi Kombinasi (Supervised + Unsupervised)**

    Menggabungkan kedua pendekatan:
    - **Supervised**: Estimasi potensi engagement dari metadata pra-tayang
    - **Unsupervised**: Klasterisasi engagement aktual dari metrik pasca-tayang
    - **Alignment Analysis**: Apakah prediksi pra-tayang selaras dengan performa aktual?
    """
    un_bundle = get_unsupervised_model()
    sup_bundle = get_supervised_model()

    try:
        un_result = predict_cluster(data.views, data.likes, data.comments, un_bundle)
        sup_result = predict_class(
            data.title, data.channel or "unknown",
            data.country or "unknown", data.published_at, sup_bundle
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediksi gagal: {str(e)}")

    supervised_class = sup_result["predicted_class"]
    unsupervised_level = un_result["engagement_level"]
    aligned = supervised_class == unsupervised_level

    alignment_score = float(sup_result["class_probabilities"].get(unsupervised_level, 0))

    return {
        "model_types": ["supervised", "unsupervised"],
        "input": {
            "title": data.title,
            "channel": data.channel,
            "country": data.country,
            "published_at": data.published_at,
            "views": data.views,
            "likes": data.likes,
            "comments": data.comments,
        },
        "supervised": {
            "model_name": sup_bundle.get("model_name"),
            "predicted_class": supervised_class,
            "confidence": sup_result["confidence"],
            "class_probabilities": sup_result["class_probabilities"],
            "description": "Estimasi potensi engagement berdasarkan metadata pra-tayang",
        },
        "unsupervised": {
            "model_name": un_bundle.get("model_name"),
            "engagement_level": unsupervised_level,
            "engagement_rate": un_result["engagement_rate"],
            "cluster_id": un_result["cluster_id"],
            "confidence_score": un_result["confidence_score"],
            "description": "Klasterisasi engagement aktual berdasarkan metrik pasca-tayang",
        },
        "alignment": {
            "is_aligned": aligned,
            "supervised_prediction": supervised_class,
            "actual_cluster": unsupervised_level,
            "alignment_probability": round(alignment_score, 4),
            "interpretation": (
                f"✅ SELARAS: Model supervised berhasil memprediksi level '{supervised_class}' "
                f"yang sesuai dengan klaster aktual '{unsupervised_level}'"
                if aligned else
                f"⚠️ TIDAK SELARAS: Model supervised memprediksi '{supervised_class}' "
                f"tapi klaster aktual adalah '{unsupervised_level}'. "
                f"Hal ini wajar karena akurasi model ~64% — ada faktor viralitas yang tidak "
                f"tertangkap dari metadata saja."
            ),
            "global_alignment_rate": "~63.85% (dari eksperimen pada data 2025)",
        }
    }


@app.post("/machine-learning/predict/youtube", tags=["Prediksi"])
async def predict_youtube(data: YouTubeUrlInput):
    """
    **Prediksi Otomatis dari Tautan / ID Video YouTube**

    Menerima tautan YouTube atau ID Video 11 karakter, mengambil metadata
    menggunakan YouTube Data API v3 (jika dikonfigurasi) atau via scraping fallback,
    kemudian melakukan prediksi Supervised + Unsupervised + Alignment secara otomatis.
    """
    video_id = extract_video_id(data.url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="Tautan atau ID Video YouTube tidak valid. Harus mengandung 11 karakter video_id."
        )

    try:
        # 1. Fetch metadata
        metadata = fetch_youtube_metadata(video_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil data video dari YouTube: {str(e)}"
        )

    title = metadata.get("title", "Unknown Video")
    channel = metadata.get("channel", "unknown")
    published_at = metadata.get("published_at")
    views = metadata.get("views", 0)
    original_likes = metadata.get("likes", 0)
    original_comments = metadata.get("comments", 0)

    # Deteksi dan tangani data cacat (bias) secara cerdas via Data Imputation
    data_imputed = False
    warning_msg = None
    likes = original_likes
    comments = original_comments

    # Jika views banyak tetapi likes 0, itu berarti data likes gagal diambil (bias scraping/API)
    if views > 100 and original_likes == 0:
        data_imputed = True
        # Lakukan estimasi rasio engagement sehat (likes ~ 2% dari views, comments ~ 0.05% dari views)
        likes = max(1, int(views * 0.02))
        comments = max(1, int(views * 0.0005))
        warning_msg = "Metrik interaksi asli (likes/comments) bernilai 0. Sistem menggunakan estimasi imputasi data (Engagement ~2.05%) untuk menghindari bias klasterisasi."

    un_bundle = get_unsupervised_model()
    sup_bundle = get_supervised_model()

    try:
        un_result = predict_cluster(views, likes, comments, un_bundle)
        sup_result = predict_class(title, channel, "unknown", published_at, sup_bundle)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediksi gagal: {str(e)}")

    supervised_class = sup_result["predicted_class"]
    unsupervised_level = un_result["engagement_level"]
    aligned = supervised_class == unsupervised_level
    alignment_score = float(sup_result["class_probabilities"].get(unsupervised_level, 0))

    return {
        "video_id": video_id,
        "metadata_method": metadata.get("method"),
        "data_imputed": data_imputed,
        "warning": warning_msg,
        "input_extracted": {
            "title": title,
            "channel": channel,
            "published_at": published_at,
            "views": views,
            "likes": original_likes,
            "comments": original_comments,
            "imputed_likes": likes if data_imputed else None,
            "imputed_comments": comments if data_imputed else None,
        },
        "supervised": {
            "model_name": sup_bundle.get("model_name"),
            "predicted_class": supervised_class,
            "confidence": sup_result["confidence"],
            "class_probabilities": sup_result["class_probabilities"],
        },
        "unsupervised": {
            "model_name": un_bundle.get("model_name"),
            "engagement_level": unsupervised_level,
            "engagement_rate": un_result["engagement_rate"],
            "confidence_score": un_result["confidence_score"],
        },
        "alignment": {
            "is_aligned": aligned,
            "supervised_prediction": supervised_class,
            "actual_cluster": unsupervised_level,
            "alignment_probability": round(alignment_score, 4),
            "interpretation": (
                f"✅ SELARAS: Model supervised memprediksi '{supervised_class}' "
                f"sesuai dengan klaster aktual '{unsupervised_level}'"
                if aligned else
                f"⚠️ TIDAK SELARAS: Model supervised memprediksi '{supervised_class}' "
                f"tapi klaster aktual adalah '{unsupervised_level}'."
            ),
        }
    }

