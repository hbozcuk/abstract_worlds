# -*- coding: utf-8 -*-
"""
Streamlit app for abstract inner vs outer worlds,
with truly on-demand SD2.1 loading (no model fetch at startup).
"""
import os
# Disable Streamlit file-watcher (avoids torch internal errors)
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon

import inspect, sys, textwrap, pathlib, re, os
if "DIFFUSERS_CHECK" not in os.environ:
    os.environ["DIFFUSERS_CHECK"] = "1"
    src = pathlib.Path(__file__).read_text().splitlines()
    offending = [f"{i:>3}: {line}" for i,line in enumerate(src,1) if re.search(r'diffusers', line)]
    print(">>> LINES CONTAINING 'diffusers' AT STARTUP:")
    print("\n".join(offending) or ">>> None <<<")


# ─────────────────────────────────────────────────────────────────────────────
# 1) Manual lazy loader — diffusers only imported here
# ─────────────────────────────────────────────────────────────────────────────
def load_pipeline():
    print(">>> DEBUG: load_pipeline() executed!")

    # now we import only when needed
    import torch
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch_dtype
    ).to(device)

    if device == "cpu":
        pipe.enable_attention_slicing()

    return pipe

# keep pipeline in session state  
if "pipe" not in st.session_state:
    st.session_state.pipe = None

# ─────────────────────────────────────────────────────────────────────────────
# 2) Metric computation (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    # HSV
    hsv = img.convert("HSV")
    h, s, v = np.array(hsv).transpose(2,0,1)
    hue_mean, sat_mean, val_mean = h.mean(), s.mean(), v.mean()
    sat_std, hue_std, val_std   = s.std(), h.std(), v.std()

    # Contrast
    gray = np.array(img.convert("L"), float)
    contrast = gray.std()

    # Colorfulness
    R, G, B = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
    rg, yb   = R - G, 0.5*(R+G) - B
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
st.set_page_config(page_title="Soyut İç & Dış Dünya", layout="wide")
st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")

inner_txt = st.text_area("📖 İç Dünya:", height=120)
outer_txt = st.text_area("🌍 Dış Dünya:", height=120)

if st.button("🎨 Oluştur ve Karşılaştır"):
    if not inner_txt or not outer_txt:
        st.warning("⚠️ Lütfen hem iç hem dış dünyayı doldurun.")
    else:
        # 1) Lazy-load the model
        if st.session_state.pipe is None:
            with st.spinner("📥 Model indiriliyor, lütfen bekleyin…"):
                st.session_state.pipe = load_pipeline()
        pipe = st.session_state.pipe

        # 2) Generate images
        with st.spinner("🖼️ Görseller üretiliyor…"):
            prompt1 = f"Soyut sanat eseri, non-figurative geometric abstraction: {inner_txt}"
            prompt2 = f"Soyut sanat eseri, non-figurative geometric abstraction: {outer_txt}"
            img1 = pipe(prompt1, num_inference_steps=50, guidance_scale=9.0).images[0]
            img2 = pipe(prompt2, num_inference_steps=50, guidance_scale=9.0).images[0]

        # 3) Display side by side
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("İç Dünya")
            st.image(img1, use_column_width=True)
        with c2:
            st.subheader("Dış Dünya")
            st.image(img2, use_column_width=True)

        # 4) Save for metric calculation
        img1.save("inner.png")
        img2.save("outer.png")

        # 5) Compute metrics
        A = np.array(compute_metrics("inner.png"))
        B = np.array(compute_metrics("outer.png"))

        # 6) IoU resemblance
        labels = [
            "Renk Tonu","Doygunluk","Parlaklık","Kontrast",
            "Renk Canlılığı","Doygunluk Sapması","Renk Tonu Sapması",
            "Parlaklık Sapması","Detay Seviyesi"
        ]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        ptsA = [(r*np.cos(a), r*np.sin(a)) for r,a in zip(np.append(A,A[0]), angles)]
        ptsB = [(r*np.cos(a), r*np.sin(a)) for r,a in zip(np.append(B,B[0]), angles)]
        polyA, polyB = Polygon(ptsA), Polygon(ptsB)
        iou = polyA.intersection(polyB).area / polyA.union(polyB).area
        st.markdown(f"**🔍 Benzerlik (IoU): {iou:.3f}**")

        # 7) Radar plot
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