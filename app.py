# -*- coding: utf-8 -*-
"""
Streamlit app – generates abstract “inner” & “outer” world images
and compares them with a radar chart and IoU score.
Loads Stable-Diffusion 2.1 **only after** the user presses a button.
"""
import os
# Disable Streamlit file-watcher (avoids torch internal errors)
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"

import streamlit as st

# ── 0) Abort early if this is just Streamlit Cloud’s health-check ────────────
if os.environ.get("ST_STATE") == "health-check":
    st.write("✅ Health-check OK – heavy code skipped.")
    st.stop()

# (Lightweight imports only – no diffusers / torch here)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon

# Global handle for the pipeline (persists across reruns in same session)
_PIPE = None   # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# 1) Lazy loader - diffusers is imported **inside** this function only
# ─────────────────────────────────────────────────────────────────────────────
def load_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    # Heavy imports happen now, not at file-top
    import torch
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    _PIPE  = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=dtype
    ).to(device)

    if device == "cpu":
        _PIPE.enable_attention_slicing()

    return _PIPE

# ─────────────────────────────────────────────────────────────────────────────
# 2) Metric computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(path: str):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    hsv = img.convert("HSV")
    h, s, v = np.array(hsv).transpose(2, 0, 1)
    hue_mean, sat_mean, val_mean = h.mean(), s.mean(), v.mean()
    sat_std, hue_std, val_std    = s.std(), h.std(), v.std()

    gray      = np.array(img.convert("L"), float)
    contrast  = gray.std()

    R, G, B   = arr[..., 0].astype(float), arr[..., 1].astype(float), arr[..., 2].astype(float)
    rg, yb    = R - G, 0.5 * (R + G) - B
    colorfulness = np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)

    gy, gx    = np.gradient(gray)
    detail    = np.mean(np.sqrt(gx**2 + gy**2))

    return [
        hue_mean, sat_mean, val_mean,
        contrast, colorfulness,
        sat_std, hue_std, val_std,
        detail,
    ]

# ─────────────────────────────────────────────────────────────────────────────
# 3) Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Soyut İç & Dış Dünya", layout="wide")
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
        prompt1 = f"Soyut sanat eseri, non-figurative geometric abstraction: {inner_txt}"
        prompt2 = f"Soyut sanat eseri, non-figurative geometric abstraction: {outer_txt}"
        img_inner = pipe(prompt1, num_inference_steps=50, guidance_scale=9.0).images[0]
        img_outer = pipe(prompt2, num_inference_steps=50, guidance_scale=9.0).images[0]

    # 3) Show images
    c1, c2 = st.columns(2)
    c1.subheader("İç Dünya");  c1.image(img_inner, use_column_width=True)
    c2.subheader("Dış Dünya"); c2.image(img_outer, use_column_width=True)

    # 4) Save temp files & compute metrics
    img_inner.save("inner_tmp.png"); img_outer.save("outer_tmp.png")
    A = np.array(compute_metrics("inner_tmp.png"))
    B = np.array(compute_metrics("outer_tmp.png"))

    # 5) IoU resemblance
    labels = [
        "Renk Tonu", "Doygunluk", "Parlaklık", "Kontrast",
        "Renk Canlılığı", "Doygunluk Sap.", "Renk Tonu Sap.",
        "Parlaklık Sap.", "Detay Seviyesi",
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    ptsA = [(r * np.cos(a), r * np.sin(a)) for r, a in zip(np.append(A, A[0]), angles)]
    ptsB = [(r * np.cos(a), r * np.sin(a)) for r, a in zip(np.append(B, B[0]), angles)]
    polyA, polyB = Polygon(ptsA), Polygon(ptsB)
    iou = polyA.intersection(polyB).area / polyA.union(polyB).area
    st.markdown(f"**🔍 Benzerlik (IoU): {iou:.3f}**")

    # 6) Radar chart
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(angles, np.append(A, A[0]), label="İç Dünya", lw=2)
    ax.fill(angles, np.append(A, A[0]), alpha=0.25)
    ax.plot(angles, np.append(B, B[0]), label="Dış Dünya", lw=2)
    ax.fill(angles, np.append(B, B[0]), alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_rlabel_position(30)
    ax.set_title("Son Görsel Metrikleri", va="bottom", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    st.pyplot(fig)