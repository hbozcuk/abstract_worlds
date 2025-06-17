# File: app.py

# -*- coding: utf-8 -*-
"""
Streamlit app – compares abstract "inner" & "outer" world images.
Uses Hugging Face Inference API to avoid heavy local model loading.
"""

# 0) Monkey-patch asyncio.get_running_loop to avoid Streamlit watcher bug
import asyncio
_orig_loop = asyncio.get_running_loop

def _safe_get_loop():
    try:
        return _orig_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

asyncio.get_running_loop = _safe_get_loop

import os
import traceback

# Configure environment for Streamlit Cloud
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

import streamlit as st
from huggingface_hub import InferenceApi
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Soyut İç & Dış Dünya", layout="wide")

# Health check handling
if os.environ.get("ST_STATE") == "health-check":
    st.stop()

# Get Hugging Face token
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)
if not HF_TOKEN:
    st.error("❌ Hugging Face token eksik! Lütfen Streamlit Secrets'a ekleyin.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_api():
    # Initialize Inference API client for a lightweight remote model
    # enable wait_for_model at client init
    return InferenceApi(repo_id="prompthero/openjourney", token=HF_TOKEN, wait_for_model=True)  :
    # Initialize Inference API client for a lightweight remote model
    return InferenceApi(repo_id="prompthero/openjourney", token=HF_TOKEN)


def compute_metrics(arr):
    """Compute basic image metrics"""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness = (r + g + b).mean() / 3
    contrast = np.std(r) + np.std(g) + np.std(b)
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gx, gy = np.gradient(gray)
    detail = np.mean(np.sqrt(gx ** 2 + gy ** 2))
    return [brightness, contrast, colorfulness, detail]


def calculate_iou(A, B):
    A_norm = (A - np.min(A)) / (np.max(A) - np.min(A) + 1e-10)
    B_norm = (B - np.min(B)) / (np.max(B) - np.min(B) + 1e-10)
    inter = np.sum(np.minimum(A_norm, B_norm))
    union = np.sum(np.maximum(A_norm, B_norm))
    return inter / union if union > 0 else 0

# Streamlit UI
st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")
inner_txt = st.text_area("📖 İç Dünya:", height=120)
outer_txt = st.text_area("🌍 Dış Dünya:", height=120)

if st.button("🎨 Oluştur ve Karşılaştır"):
    if not inner_txt or not outer_txt:
        st.warning("⚠️ Lütfen her iki metni de girin.")
    else:
        api = load_api()
        with st.spinner("🖼️ Görseller üretiliyor…"):
            # Remote inference: returns a PIL Image or list of PIL Images
            imgs1 = api(f"mdjrny-v4 style abstract art: {inner_txt}")
            imgs2 = api(f"mdjrny-v4 style abstract art: {outer_txt}")
            # API may return a single PIL Image or a list
            img1 = imgs1[0] if isinstance(imgs1, list) else imgs1
            img2 = imgs2[0] if isinstance(imgs2, list) else imgs2[0] if isinstance(imgs2, list) else imgs2

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("İç Dünya")
            st.image(img1, use_column_width=True)
        with col2:
            st.subheader("Dış Dünya")
            st.image(img2, use_column_width=True)

        arr1, arr2 = np.array(img1), np.array(img2)
        m1, m2 = compute_metrics(arr1), compute_metrics(arr2)
        iou = calculate_iou(np.array(m1), np.array(m2))
        st.markdown(f"**🔍 Benzerlik Oranı: {iou:.3f}**")

        labels = ["Parlaklık", "Kontrast", "Renk Canlılığı", "Detay"]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        m1 += m1[:1]
        m2 += m2[:1]
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"polar": True})
        ax.plot(angles, m1, "o-", label="İç Dünya"); ax.fill(angles, m1, alpha=0.25)
        ax.plot(angles, m2, "o-", label="Dış Dünya"); ax.fill(angles, m2, alpha=0.25)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
        ax.set_title("Görsel Metrik Karşılaştırması")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)