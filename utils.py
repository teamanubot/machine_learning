import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CategoryTargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder untuk fitur kategoris tanpa data leakage."""

    def __init__(self, cols=None, classes=None, smoothing=50.0):
        self.cols = cols
        self.classes = classes
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.DataFrame(X, columns=self.cols).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self.classes_ = np.array(
            self.classes if self.classes is not None else sorted(y.unique())
        )
        global_counts = y.value_counts(normalize=True)
        self.global_probs_ = np.array(
            [global_counts.get(cls, 0.0) for cls in self.classes_], dtype=float
        )
        self.maps_ = {}
        for col in X.columns:
            data = pd.DataFrame(
                {"category": X[col].fillna("unknown").astype(str), "target": y}
            )
            counts = (
                data.groupby("category")["target"]
                .value_counts()
                .unstack(fill_value=0)
            )
            counts = counts.reindex(columns=self.classes_, fill_value=0)
            n = counts.sum(axis=1).astype(float)
            probs = (counts + self.smoothing * self.global_probs_).div(
                n + self.smoothing, axis=0
            )
            probs["__count_log__"] = np.log1p(n)
            self.maps_[col] = probs
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cols)
        default = np.r_[self.global_probs_, 0.0]
        blocks = []
        for col in X.columns:
            encoded = (
                self.maps_[col]
                .reindex(X[col].fillna("unknown").astype(str))
                .to_numpy(dtype=float)
            )
            missing = np.isnan(encoded).any(axis=1)
            if missing.any():
                encoded[missing] = default
            blocks.append(encoded)
        return np.hstack(blocks) if blocks else np.empty((len(X), 0))


import re
import os
import requests
import json
from typing import Optional

def extract_video_id(url: str) -> Optional[str]:
    """Ekstraksi video_id 11 karakter dari URL YouTube."""
    if len(url) == 11 and re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_youtube_metadata(video_id: str) -> dict:
    """Mengambil metadata video YouTube dari API Resmi v3 (jika ada key) atau via scraping fallback."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    
    # Percobaan 1: Menggunakan YouTube Data API v3
    if api_key:
        try:
            url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={api_key}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("items"):
                    item = data["items"][0]
                    snippet = item.get("snippet", {})
                    stats = item.get("statistics", {})
                    channel_id = snippet.get("channelId")
                    
                    subscribers = 0
                    total_videos = 0
                    
                    # Ambil statistik channel jika channel_id ada
                    if channel_id:
                        chan_url = f"https://www.googleapis.com/youtube/v3/channels?part=statistics&id={channel_id}&key={api_key}"
                        chan_resp = requests.get(chan_url, timeout=10)
                        if chan_resp.status_code == 200:
                            chan_data = chan_resp.json()
                            if chan_data.get("items"):
                                chan_stats = chan_data["items"][0].get("statistics", {})
                                subscribers = int(chan_stats.get("subscriberCount", 0))
                                total_videos = int(chan_stats.get("videoCount", 0))
                    
                    return {
                        "title": snippet.get("title", ""),
                        "description": snippet.get("description", ""),
                        "channel": snippet.get("channelTitle", "unknown"),
                        "published_at": snippet.get("publishedAt", ""),
                        "views": int(stats.get("viewCount", 0)),
                        "likes": int(stats.get("likeCount", 0)),
                        "comments": int(stats.get("commentCount", 0)),
                        "channel_subscribers": subscribers,
                        "channel_total_videos": total_videos,
                        "method": "official_api"
                    }
        except Exception as e:
            # Jika gagal, fallback ke scraping
            pass

    # Percobaan 2: Fallback dengan scraping halaman video YouTube (tidak butuh API key)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code != 200:
        raise Exception(f"Gagal memuat halaman YouTube: HTTP {resp.status_code}")
        
    html = resp.text
    
    # Ekstraksi meta tags
    title_match = re.search(r'<meta name="title" content="([^"]+)">', html)
    title = title_match.group(1) if title_match else None
    if not title:
        title_match = re.search(r'<meta property="og:title" content="([^"]+)">', html)
        title = title_match.group(1) if title_match else "Unknown Video"
        
    desc_match = re.search(r'<meta name="description" content="([^"]+)">', html)
    description = desc_match.group(1) if desc_match else ""
    
    pub_match = re.search(r'<meta itemprop="datePublished" content="([^"]+)">', html)
    if not pub_match:
        pub_match = re.search(r'<meta itemprop="uploadDate" content="([^"]+)">', html)
    published_at = pub_match.group(1) if pub_match else None
    
    views_match = re.search(r'<meta itemprop="interactionCount" content="([^"]+)">', html)
    views = int(views_match.group(1)) if views_match else 0
    
    channel_match = re.search(r'<link itemprop="name" content="([^"]+)">', html)
    channel = channel_match.group(1) if channel_match else "unknown"
    
    likes = 0
    comments = 0
    
    # Coba parser ytInitialPlayerResponse untuk likes/views yang lebih akurat
    try:
        player_resp_match = re.search(r'ytInitialPlayerResponse\s*=\s*({.+?});', html)
        if player_resp_match:
            player_data = json.loads(player_resp_match.group(1))
            video_details = player_data.get("videoDetails", {})
            title = video_details.get("title", title)
            channel = video_details.get("author", channel)
            views = int(video_details.get("viewCount", views))
    except Exception:
        pass
        
    # Optimasi Scraping Fallback: Cari likes dari html (ytInitialData atau text label)
    try:
        likes_patterns = [
            r'"accessibilityData"\s*:\s*\{\s*"label"\s*:\s*"([0-9.,\s\xa0]+)\s*likes?"',
            r'"Like this video along with ([0-9.,\s\xa0]+) other',
            r'"label"\s*:\s*"Like this video along with ([0-9.,\s\xa0]+) other',
            r'"label"\s*:\s*"([0-9.,\s\xa0]+)\s*likes"',
            r'([0-9.,\s\xa0]+)\s*likes\b',
            r'"label"\s*:\s*"menyukai video ini bersama ([0-9.,\s\xa0]+) orang'
        ]
        for pat in likes_patterns:
            m = re.search(pat, html, re.IGNORECASE)
            if m:
                val = m.group(1).replace(",", "").replace(".", "").replace(" ", "").replace("\xa0", "").strip()
                if val.isdigit():
                    likes = int(val)
                    break
    except Exception:
        pass

    # Optimasi Scraping Fallback: Cari comments count dari html
    try:
        comments_patterns = [
            r'"commentCount"\s*:\s*\{\s*"simpleText"\s*:\s*"([0-9.,\s\xa0]+)"',
            r'"accessibilityData"\s*:\s*\{\s*"label"\s*:\s*"([0-9.,\s\xa0]+)\s*Comments?"',
            r'"accessibilityData"\s*:\s*\{\s*"label"\s*:\s*"([0-9.,\s\xa0]+)\s*Komentar?"',
            r'([0-9.,\s\xa0]+)\s*comments\b'
        ]
        for pat in comments_patterns:
            m = re.search(pat, html, re.IGNORECASE)
            if m:
                val = m.group(1).replace(",", "").replace(".", "").replace(" ", "").replace("\xa0", "").strip()
                if val.isdigit():
                    comments = int(val)
                    break
    except Exception:
        pass
        
    return {
        "title": title,
        "description": description,
        "channel": channel,
        "published_at": published_at,
        "views": views,
        "likes": likes,
        "comments": comments,
        "channel_subscribers": 0,      # Nilai default untuk scraping fallback
        "channel_total_videos": 0,     # Nilai default untuk scraping fallback
        "method": "scraping_fallback"
    }

