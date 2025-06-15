# -*- coding: utf-8 -*-
"""
Streamlit app – compares “inner” & “outer” world images.
The Stable-Diffusion 2.1 model downloads **only after** the user clicks
the button, so the Streamlit-Cloud health-check is lightning-fast.
"""

##############################################################################
# 0) Absolutely minimal Streamlit setup – page config MUST be first
##############################################################################
import os
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"      # silence watcher

import streamlit as st
st.set_page_config(page_title="Soyut İç & Dış Dünya", layout="wide")

# ── Abort instantly if this is just Streamlit Cloud’s health probe ──────────
if os.environ.get("ST_STATE") == "health-check":  # Cloud sets this env-var
    st.stop()

##############################################################################
# 1) Lightweight standard-library / NumPy imports (safe for fast startup)
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon

##############################################################################
# 2) Lazy loader – diffusers / torch imported ONLY here
##############################################################################
_PIPE = None  # global cache inside the process


def load_pipeline():
    """
    Create (or return) the Stable-Diffusion 2.1 pipeline.
    Heavy imports happen here, never at file-top.
    """
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    # heavy imports NOW:
    import torch
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    _PIPE = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=dtype,
    ).to(device)

    if device == "cpu":
        _PIPE.enable_attention_slicing()

    return _PIPE


##############################################################################
# 3) Metric computation helper
##############################################################################
def compute_metrics(path: str):
    """
    Return 9 visual metrics for a saved image.
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    hsv = img.convert("HSV")
    h, s, v = np.array(hsv).transpose(2, 0, 1)
    hue_mean, sat_mean, val_mean = h.mean(), s.mean(), v.mean()
    sat_std, hue_std, val_std = s.std(), h.std(), v.std()

    gray = np.array(img.convert("L"), float)
    contrast = gray.std()

    R, G, B = arr[..., 0].astype(float), arr[..., 1].astype(float), arr[..., 2].astype(float)
    rg, yb = R - G, 0.5 * (R + G) - B
    colorfulness = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(
        rg.mean() ** 2 + yb.mean() ** 2
    )

    gy, gx = np.gradient(gray)
    detail = np.mean(np.sqrt(gx**2 + gy**2))

    return [
        hue_mean,
        sat_mean,
        val_mean,
        contrast,
        colorfulness,
        sat_std,
        hue_std,
        val_std,
        detail,
    ]


##############################################################################
# 4) Streamlit UI
##############################################################################
st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")

inner_txt = st.text_area("📖 İç Dünya:", height=120)
outer_txt = st.text_area("🌍 Dış Dünya:", height=120)

if st.button("🎨 Oluştur ve Karşılaştır"):
    if not inner_txt or not outer_txt:
        st.warning("⚠️ Lütfen hem iç hem dış dünya metinlerini girin.")
        st.stop()

    # 1) Load model on demand
    with st.spinner("📥 Model indiriliyor / yükleniyor…"):
        pipe = load_pipeline()

    # 2) Generate images
    with st.spinner("🖼️ Görseller üretiliyor…"):
        p_inner = f"Soyut sanat eseri, non-figurative geometric abstraction: {inner_txt}"
        p_outer = f"Soyut sanat eseri, non-figurative geometric abstraction: {outer_txt}"
        img_inner = pipe(p_inner, num_inference_steps=50, guidance_scale=9.0).images[0]
        img_outer = pipe(p_outer, num_inference_steps=50, guidance_scale=9.0).images[0]

    # 3) Display images
    col1, col2 = st.columns(2)
    col1.subheader("İç Dünya");  col1.image(img_inner, use_column_width=True)
    col2.subheader("Dış Dünya"); col2.image(img_outer, use_column_width=True)

    # 4) Metrics
    img_inner.save("inner_tmp.png"); img_outer.save("outer_tmp.png")
    A = np.array(compute_metrics("inner_tmp.png"))
    B = np.array(compute_metrics("outer_tmp.png"))

    # 5) IoU resemblance
    labels = [
        "Renk Tonu","Doygunluk","Parlaklık","Kontrast",
        "Renk Canlılığı","Doygunluk Sapması","Renk Tonu Sapması",
        "Parlaklık Sapması","Detay Seviyesi"
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    ptsA = [(r * np.cos(a), r * np.sin(a)) for r, a in zip(np.append(A, A[0]), angles)]
    ptsB = [(r * np.cos(a), r * np.sin(a)) for r, a in zip(np.append(B, B[0]), angles)]
    iou = Polygon(ptsA).intersection(Polygon(ptsB)).area / Polygon(ptsA).union(Polygon(ptsB)).area
    st.markdown(f"**🔍 Benzerlik (IoU): {iou:.3f}**")

    # 6) Radar chart
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(angles, np.append(A, A[0]), label="İç Dünya", lw=2)
    ax.fill(angles, np.append(A, A[0]), alpha=0.25)
    ax.plot(angles, np.append(B, B[0]), label="Dış Dünya", lw=2)
    ax.fill(angles, np.append(B, B[0]), alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_rlabel_position(30)