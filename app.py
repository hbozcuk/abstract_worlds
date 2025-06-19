# -*- coding: utf-8 -*-
"""
Streamlit app – compares abstract "inner" & "outer" world images using free DALL·E mini
"""

# 1) Imports & env config
import os
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import io
import time
from transformers import pipeline

# 2) Page config
st.set_page_config(page_title="Soyut İç & Dış Dünya", layout="wide")

# 3) Utility functions
def compute_metrics(arr: np.ndarray):
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
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

def generate_image_dalle(prompt: str):
    """Generate image using free DALL·E mini API"""
    try:
        API_URL = "https://bf.dallemini.ai/generate"
        response = requests.post(API_URL, json={"prompt": prompt})
        response.raise_for_status()
        image_data = response.json()["images"][0]
        image_bytes = io.BytesIO(bytes(image_data, "utf-8"))
        return Image.open(image_bytes).convert("RGB")
    except Exception as err:
        st.error(f"❌ Görsel oluşturma hatası: {str(err)}")
        return None

# 4) UI
st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")
st.info("ℹ️ Ücretsiz DALL·E mini API kullanılıyor. Görsel kalitesi sınırlı olabilir")

inner_txt = st.text_area("📖 İç Dünya:", height=120, value="Rüyalarımda gördüğüm renkli dünya")
outer_txt = st.text_area("🌍 Dış Dünya:", height=120, value="Şehirdeki gri binalar ve trafik")

if st.button("🎨 Oluştur ve Karşılaştır"):
    if not inner_txt or not outer_txt:
        st.warning("⚠️ Lütfen her iki metni de girin.")
        st.stop()

    with st.spinner("🖼️ Görseller üretiliyor (10-20 saniye sürebilir)…"):
        img1 = generate_image_dalle(inner_txt)
        time.sleep(2)  # Avoid rate limiting
        img2 = generate_image_dalle(outer_txt)
        
        if img1 is None or img2 is None:
            st.error("❌ Görsel oluşturulamadı. Lütfen farklı metinlerle tekrar deneyin.")
            st.stop()

    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("İç Dünya")
        st.image(img1, use_column_width=True)
    with col2:
        st.subheader("Dış Dünya")
        st.image(img2, use_column_width=True)

    # Compute metrics
    try:
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        m1 = compute_metrics(arr1)
        m2 = compute_metrics(arr2)
        iou = calculate_iou(m1, m2)
        st.success(f"**🔍 Benzerlik Oranı: {iou:.3f}**")

        # Radar chart
        labels = ["Parlaklık","Kontrast","Renk Canlılığı","Detay"]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        m1_plot = np.concatenate([m1, [m1[0]]])
        m2_plot = np.concatenate([m2, [m2[0]]])

        fig, ax = plt.subplots(figsize=(6,6), subplot_kw={"polar":True})
        ax.plot(angles, m1_plot, 'o-', label='İç Dünya')
        ax.fill(angles, m1_plot, alpha=0.25)
        ax.plot(angles, m2_plot, 'o-', label='Dış Dünya')
        ax.fill(angles, m2_plot, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title("Görsel Metrik Karşılaştırması", va='bottom')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"❌ Analiz hatası: {str(e)}")