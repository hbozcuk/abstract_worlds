# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 10:10:58 2025

@author: hbozc
"""

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 1) Monkey-patch for cached_download must come before diffusers import!
# ─────────────────────────────────────────────────────────────────────────────
import huggingface_hub as _hf
_hf.cached_download = _hf.hf_hub_download

# ─────────────────────────────────────────────────────────────────────────────
# 2) Now import the rest
# ─────────────────────────────────────────────────────────────────────────────
import torch, io, numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from diffusers import StableDiffusionPipeline
from shapely.geometry import Polygon

# ─────────────────────────────────────────────────────────────────────────────
# 3) Cache the Stable-Diffusion pipeline
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    ).to(device)
    return pipe

pipe = load_pipe()

# ─────────────────────────────────────────────────────────────────────────────
# Compute the 9 metrics for a saved image path
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
    rg = R-G; yb = 0.5*(R+G)-B
    std_rg,std_yb = rg.std(), yb.std()
    mean_rg,mean_yb = rg.mean(), yb.mean()
    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3*np.sqrt(mean_rg**2 + mean_yb**2)
    # Detail (gradient magnitude)
    gy, gx = np.gradient(gray)
    detail = np.mean(np.sqrt(gx**2+gy**2))
    return [hue_mean, sat_mean, val_mean,
            contrast, colorfulness,
            sat_std, hue_std, val_std,
            detail]

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")

inner_txt = st.text_area("İç Dünya:", height=100)
outer_txt = st.text_area("Dış Dünya:", height=100)

if st.button("Oluştur ve Karşılaştır"):
    if not inner_txt or not outer_txt:
        st.warning("⚠️ Lütfen her iki metni de girin.")
    else:
        with st.spinner("Görseller üretiliyor…"):
            # prompts
            p1 = (f"Soyut sanat eseri, non-figurative geometric abstraction: "
                  f"{inner_txt}. Organik soyut formlar, sürreal renk kompozisyonu, 8k.")
            p2 = (f"Soyut sanat eseri, non-figurative geometric abstraction: "
                  f"{outer_txt}. Organik soyut formlar, sürreal renk kompozisyonu, 8k.")
            img1 = pipe(p1, num_inference_steps=50, guidance_scale=9.0).images[0]
            img2 = pipe(p2, num_inference_steps=50, guidance_scale=9.0).images[0]

        # display images
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("İç Dünya Görseli")
            st.image(img1, use_column_width=True)
        with col2:
            st.subheader("Dış Dünya Görseli")
            st.image(img2, use_column_width=True)

        # save
        img1.save("inner_abstract.png")
        img2.save("outer_abstract.png")

        # compute metrics
        A = np.array(compute_metrics("inner_abstract.png"))
        B = np.array(compute_metrics("outer_abstract.png"))

        # build polygons
        labels = [
          "Renk Tonu","Doygunluk","Parlaklık",
          "Kontrast","Renk Canlılığı",
          "Doygunluk Sapması","Renk Tonu Sapması","Parlaklık Sapması",
          "Detay Seviyesi"
        ]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        ptsA = [(r*np.cos(a), r*np.sin(a)) for r,a in zip(np.append(A,A[0]), angles)]
        ptsB = [(r*np.cos(a), r*np.sin(a)) for r,a in zip(np.append(B,B[0]), angles)]
        polyA, polyB = Polygon(ptsA), Polygon(ptsB)

        # IoU
        inter = polyA.intersection(polyB).area
        union = polyA.union(polyB).area
        iou   = inter/union if union>0 else 0.0
        st.write(f"**Benzerlik (IoU) = {iou:.3f}**")

        # radar chart
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
