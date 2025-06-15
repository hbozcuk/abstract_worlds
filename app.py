# app.py
# -*- coding: utf-8 -*-
"""
Streamlit app for generating and comparing abstract “inner” vs “outer” world images
with radar‐chart metrics and IoU resemblance score.
"""
import os
# Disable Streamlit’s file watcher to avoid introspecting torch internals
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline
from shapely.geometry import Polygon

# ─────────────────────────────────────────────────────────────────────────────
# 1) Lazy‐loaded, cached Stable Diffusion loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch_dtype
    ).to(device)

    if device == "cpu":
        pipe.enable_attention_slicing()

    return pipe

# ensure we only load the pipeline once after user interaction
if "pipe" not in st.session_state:
    st.session_state.pipe = None

# ─────────────────────────────────────────────────────────────────────────────
# 2) Metric computation function
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    # HSV statistics
    hsv = img.convert("HSV")
    h, s, v = np.array(hsv).transpose(2,0,1)
    hue_mean, sat_mean, val_mean = h.mean(), s.mean(), v.mean()
    sat_std, hue_std, val_std = s.std(), h.std(), v.std()

    # Contrast (grayscale standard deviation)
    gray = np.array(img.convert("L"), dtype=float)
    contrast = gray.std()

    # Colorfulness (Hasler–Süsstrunk metric)
    R, G, B = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
    rg = R - G
    yb = 0.5*(R + G) - B
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

    # Detail level (mean gradient magnitude)
    gy, gx = np.gradient(gray)
    detail = np.mean(np.sqrt(gx**2 + gy**2))

    return [
        hue_mean, sat_mean, val_mean,
        contrast, colorfulness,
        sat_std, hue_std, val_std,
        detail
    ]

# ─────────────────────────────────────────────────────────────────────────────
# 3) Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Soyut İç & Dış Dünya", layout="wide")
st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")

inner_txt = st.text_area("📖 İç Dünya:", height=120, placeholder="İç dünyanızı Türkçe yazın…")
outer_txt = st.text_area("🌍 Dış Dünya:", height=120, placeholder="Dış dünyayı Türkçe yazın…")

if st.button("🎨 Oluştur ve Karşılaştır"):
    if not inner_txt or not outer_txt:
        st.warning("⚠️ Lütfen her iki metni de girin.")
    else:
        # 3a) Lazy‐load model on first use
        if st.session_state.pipe is None:
            with st.spinner("📥 Model yükleniyor, lütfen bekleyin…"):
                st.session_state.pipe = load_pipe()
        pipe = st.session_state.pipe

        # 3b) Generate images
        with st.spinner("🖼️ Görseller üretiliyor…"):
            p1 = (f"Soyut sanat eseri, non-figurative geometric abstraction: "
                  f"{inner_txt}. Organik soyut formlar, sürreal renk kompozisyonu, 8k.")
            p2 = (f"Soyut sanat eseri, non-figurative geometric abstraction: "
                  f"{outer_txt}. Organik soyut formlar, sürreal renk kompozisyonu, 8k.")
            img1 = pipe(p1, num_inference_steps=50, guidance_scale=9.0).images[0]
            img2 = pipe(p2, num_inference_steps=50, guidance_scale=9.0).images[0]

        # 3c) Display side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("İç Dünya Görseli")
            st.image(img1, use_column_width=True)
        with col2:
            st.subheader("Dış Dünya Görseli")
            st.image(img2, use_column_width=True)

        # 3d) Save locally for metrics
        img1.save("inner_abstract.png")
        img2.save("outer_abstract.png")

        # 3e) Compute metrics
        A = np.array(compute_metrics("inner_abstract.png"))
        B = np.array(compute_metrics("outer_abstract.png"))

        # 3f) IoU resemblance score
        labels = [
            "Renk Tonu","Doygunluk","Parlaklık",
            "Kontrast","Renk Canlılığı",
            "Doygunluk Sapması","Renk Tonu Sapması","Parlaklık Sapması",
            "Detay Seviyesi"
        ]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        ptsA = [(r*np.cos(a), r*np.sin(a)) for r,a in zip(np.append(A, A[0]), angles)]
        ptsB = [(r*np.cos(a), r*np.sin(a)) for r,a in zip(np.append(B, B[0]), angles)]
        polyA, polyB = Polygon(ptsA), Polygon(ptsB)
        inter = polyA.intersection(polyB).area
        union = polyA.union(polyB).area
        iou = inter/union if union>0 else 0.0

        st.markdown(f"**🔍 Benzerlik (IoU): {iou:.3f}**")

        # 3g) Radar chart of last metrics
        fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(6,6))
        ax.plot(angles, np.append(A, A[0]), label="İç Dünya", linewidth=2)
        ax.fill(angles, np.append(A, A[0]), alpha=0.25)
        ax.plot(angles, np.append(B, B[0]), label="Dış Dünya", linewidth=2)
        ax.fill(angles, np.append(B, B[0]), alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_rlabel_position(30)
        ax.set_title("Son Görsel Metrikleri", va='bottom', pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))

        plt.tight_layout()
        st.pyplot(fig)