# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 10:10:58 2025
@author: hbozc
"""

import os
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from diffusers import StableDiffusionPipeline
from shapely.geometry import Polygon

# ─────────────────────────────────────────────────────────────────────────────
# 1) Define & cache loader (but don't call at import)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device=="cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch_dtype
    ).to(device)

    if device=="cpu":
        pipe.enable_attention_slicing()

    return pipe

# Ensure session_state has a slot for our pipe
if "pipe" not in st.session_state:
    st.session_state.pipe = None

# ─────────────────────────────────────────────────────────────────────────────
# 2) Metric function (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    # HSV
    hsv = img.convert("HSV")
    h,s,v = np.array(hsv).transpose(2,0,1)
    hue_mean, sat_mean, val_mean = h.mean(), s.mean(), v.mean()
    sat_std, hue_std, val_std   = s.std(), h.std(), v.std()
    # Contrast
    gray = np.array(img.convert("L"), dtype=float)
    contrast = gray.std()
    # Colorfulness
    R,G,B = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
    rg, yb = R-G, 0.5*(R+G)-B
    colorfulness = (
        np.sqrt(rg.std()**2 + yb.std()**2)
        + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)
    )
    # Detail
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
st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")

inner_txt = st.text_area("İç Dünya:", height=100)
outer_txt = st.text_area("Dış Dünya:", height=100)

if st.button("Oluştur ve Karşılaştır"):
    # 3a) Lazy‐load the pipeline
    if st.session_state.pipe is None:
        with st.spinner("Model yükleniyor, lütfen bekleyin…"):
            st.session_state.pipe = load_pipe()

    pipe = st.session_state.pipe

    if not inner_txt or not outer_txt:
        st.warning("⚠️ Lütfen her iki metni de girin.")
    else:
        with st.spinner("Görseller üretiliyor…"):
            p1 = (
                f"Soyut sanat eseri, non-figurative geometric abstraction: "
                f"{inner_txt}. Organik soyut formlar, sürreal renk kompozisyonu, 8k."
            )
            p2 = (
                f"Soyut sanat eseri, non-figurative geometric abstraction: "
                f"{outer_txt}. Organik soyut formlar, sürreal renk kompozisyonu, 8k."
            )
            img1 = pipe(p1, num_inference_steps=50, guidance_scale=9.0).images[0]
            img2 = pipe(p2, num_inference_steps=50, guidance_scale=9.0).images[0]

        # Display
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("İç Dünya Görseli")
            st.image(img1, use_column_width=True)
        with c2:
            st.subheader("Dış Dünya Görseli")
            st.image(img2, use_column_width=True)

        # Save
        img1.save("inner_abstract.png")
        img2.save("outer_abstract.png")

        # Metrics
        A = np.array(compute_metrics("inner_abstract.png"))
        B = np.array(compute_metrics("outer_abstract.png"))

        # IoU
        labels = [
            "Renk Tonu","Doygunluk","Parlaklık",
            "Kontrast","Renk Canlılığı",
            "Doygunluk Sapması","Renk Tonu Sapması","Parlaklık Sapması",
            "Detay Seviyesi"
        ]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles,[angles[0]]))
        ptsA = [(r*np.cos(a), r*np.sin(a)) for r,a in zip(np.append(A,A[0]), angles)]
        ptsB = [(r*np.cos(a), r*np.sin(a)) for r,a in zip(np.append(B,B[0]), angles)]
        polyA, polyB = Polygon(ptsA), Polygon(ptsB)
        inter, uni = polyA.intersection(polyB).area, polyA.union(polyB).area
        iou = inter/uni if uni>0 else 0.0
        st.write(f"**Benzerlik (IoU) = {iou:.3f}**")

        # Radar
        fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(6,6))
        ax.plot(angles, np.append(A,A[0]), label="İç Dünya", linewidth=2)
        ax.fill(angles, np.append(A,A[0]), alpha=0.25)
        ax.plot(angles, np.append(B,B[0]), label="Dış Dünya", linewidth=2)
        ax.fill(angles, np.append(B,B[0]), alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_rlabel_position(30)
        ax.set_title("Son Görsel Metrikleri", va='bottom', pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))
        plt.tight_layout()
        st.pyplot(fig)