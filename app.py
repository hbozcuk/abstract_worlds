# -*- coding: utf-8 -*-
"""
Streamlit app – compares abstract "inner" & "outer" world images using remote inference.
"""

# 0) Fix asyncio watcher bug
import asyncio
_orig = asyncio.get_running_loop

def _safe_loop():
    try:
        return _orig()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
asyncio.get_running_loop = _safe_loop

# 1) Imports & env config
import os
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import InferenceClient

# 2) Page config & token
st.set_page_config(page_title="Soyut İç & Dış Dünya", layout="wide")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    st.error("❌ Hugging Face token eksik! Lütfen Streamlit Secrets'a ekleyin.")
    st.stop()

# 3) Initialize client inside handler for faster startup

# 4) Utility functions
def compute_metrics(arr: np.ndarray):
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
    brightness = (r + g + b).mean() / 3
    contrast = np.std(r) + np.std(g) + np.std(b)
    rg, yb = r - g, 0.5*(r+g) - b
    colorfulness = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gx, gy = np.gradient(gray)
    detail = np.mean(np.sqrt(gx**2 + gy**2))
    return [brightness, contrast, colorfulness, detail]

def calculate_iou(A, B):
    A, B = np.array(A), np.array(B)
    A_norm = (A - A.min()) / (A.max() - A.min() + 1e-10)
    B_norm = (B - B.min()) / (B.max() - B.min() + 1e-10)
    return np.sum(np.minimum(A_norm, B_norm)) / np.sum(np.maximum(A_norm, B_norm))

# 5) UI
st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")
inner_txt = st.text_area("📖 İç Dünya:", height=120)
outer_txt = st.text_area("🌍 Dış Dünya:", height=120)

if st.button("🎨 Oluştur ve Karşılaştır"):
    if not inner_txt or not outer_txt:
        st.warning("⚠️ Lütfen her iki metni de girin.")
        st.stop()

    client = InferenceClient(model="prompthero/openjourney", token=HF_TOKEN)
    with st.spinner("🖼️ Görseller üretiliyor…"):
        try:
            img1 = client.text_to_image(
                prompt=f"mdjrny-v4 style abstract art: {inner_txt}"
            )
            img2 = client.text_to_image(
                prompt=f"mdjrny-v4 style abstract art: {outer_txt}"
            )
        except Exception as e:
            st.error(f"❌ Görsel oluşturulurken bir hata oluştu: {e}")
            st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("İç Dünya")
        st.image(img1, use_column_width=True)
    with col2:
        st.subheader("Dış Dünya")
        st.image(img2, use_column_width=True)

    m1, m2 = compute_metrics(np.array(img1)), compute_metrics(np.array(img2))
    iou = calculate_iou(m1, m2)
    st.markdown(f"**🔍 Benzerlik Oranı: {iou:.3f}**")

    labels = ["Parlaklık","Kontrast","Renk Canlılığı","Detay"]
    angles = np.linspace(0,2*np.pi,len(labels),endpoint=False)
    angles = np.concatenate([angles,[angles[0]]])
    m1 += [m1[0]]; m2 += [m2[0]]

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw={"polar":True})
    ax.plot(angles, m1, 'o-', label='İç Dünya'); ax.fill(angles, m1, alpha=0.25)
    ax.plot(angles, m2, 'o-', label='Dış Dünya'); ax.fill(angles, m2, alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_title("Görsel Metrik Karşılaştırması", va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
    st.pyplot(fig)