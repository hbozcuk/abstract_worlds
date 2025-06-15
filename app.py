# -*- coding: utf-8 -*-
"""
Streamlit app – compares abstract "inner" & "outer" world images.
Updated deployment-friendly version without Shapely or cv2
"""

import os, sys, traceback

# Silence Streamlit's file-watcher
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Added for tokenizer safety

import streamlit as st
# 1) Must be first
st.set_page_config(page_title="Soyut İç & Dış Dünya", layout="wide")

# Abort immediately during health‐check
if os.environ.get("ST_STATE") == "health-check":
    st.stop()

# --- fetch your HF token from Streamlit Secrets ---
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)

# Check if token is available
if not HF_TOKEN:
    st.error("❌ Hugging Face token is missing! Add it to Streamlit Secrets.")
    st.stop()

try:
    ##############################################################################
    # Lightweight imports
    ##############################################################################
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    from diffusers import StableDiffusionPipeline

    ##############################################################################
    # Lazy loader that uses your HF_TOKEN
    ##############################################################################
    _PIPE = None

    @st.cache_resource(show_spinner=False)
    def load_pipeline():
        global _PIPE
        if _PIPE is not None:
            return _PIPE

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32

        # Use safer pipeline initialization
        _PIPE = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            torch_dtype=dtype,
            use_auth_token=HF_TOKEN,
            safety_checker=None  # Disable safety checker for deployment
        ).to(device)

        if device == "cpu":
            _PIPE.enable_attention_slicing()
            
        return _PIPE

    ##############################################################################
    # Metric computation
    ##############################################################################
    def compute_metrics(arr):
        img = Image.fromarray(arr).convert("RGB")
        hsv = img.convert("HSV")
        h, s, v = np.array(hsv).transpose(2,0,1)
        hue_mean, sat_mean, val_mean = h.mean(), s.mean(), v.mean()
        sat_std, hue_std, val_std    = s.std(), h.std(), v.std()
        gray = np.array(img.convert("L"), float)
        contrast = gray.std()
        R,G,B = arr[...,0].astype(float), arr[...,1].astype(float), arr[...,2].astype(float)
        rg, yb = R-G, 0.5*(R+G)-B
        colorfulness = np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)
        
        # Calculate detail without cv2
        gy, gx = np.gradient(gray)
        detail = np.mean(np.sqrt(gx**2 + gy**2))
        
        return [
            hue_mean, sat_mean, val_mean,
            contrast, colorfulness,
            sat_std, hue_std, val_std,
            detail
        ]

    ##############################################################################
    # Simplified IoU calculation
    ##############################################################################
    def calculate_iou(A, B):
        """Calculate Intersection over Union using min/max approach"""
        # Normalize metrics to same scale
        A_norm = (A - np.min(A)) / (np.max(A) - np.min(A) + 1e-10)
        B_norm = (B - np.min(B)) / (np.max(B) - np.min(B) + 1e-10)
        
        # Calculate intersection and union
        intersection = np.sum(np.minimum(A_norm, B_norm))
        union = np.sum(np.maximum(A_norm, B_norm))
        
        # Avoid division by zero
        return intersection / union if union > 0 else 0

    ##############################################################################
    # Streamlit UI
    ##############################################################################
    st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")

    inner_txt = st.text_area("📖 İç Dünya:",  height=120, value="Duygularım, düşüncelerim, içsel yolculuğum")
    outer_txt = st.text_area("🌍 Dış Dünya:", height=120, value="Sosyal ilişkiler, doğa, şehir yaşamı")

    if st.button("🎨 Oluştur ve Karşılaştır"):
        if not inner_txt or not outer_txt:
            st.warning("⚠️ Lütfen her iki metni de girin.")
        else:
            with st.spinner("📥 Model yükleniyor…"):
                pipe = load_pipeline()

            with st.spinner("🖼️ Görseller üretiliyor…"):
                p1 = f"Soyut sanat eseri, non-figurative geometric abstraction: {inner_txt}"
                p2 = f"Soyut sanat eseri, non-figurative geometric abstraction: {outer_txt}"
                
                # Generate images with fixed seed for reproducibility
                generator = torch.Generator(device=pipe.device).manual_seed(42)
                img1 = pipe(p1, generator=generator, num_inference_steps=50, guidance_scale=9.0).images[0]
                img2 = pipe(p2, generator=generator, num_inference_steps=50, guidance_scale=9.0).images[0]

            col1, col2 = st.columns(2)
            col1.subheader("İç Dünya")  
            col1.image(img1, use_column_width=True)
            col2.subheader("Dış Dünya") 
            col2.image(img2, use_column_width=True)

            # Convert to arrays and compute metrics
            arr1 = np.array(img1)
            arr2 = np.array(img2)
            A = np.array(compute_metrics(arr1))
            B = np.array(compute_metrics(arr2))

            # Simplified IoU calculation
            iou = calculate_iou(A, B)
            st.markdown(f"**🔍 Benzerlik (IoU): {iou:.3f}**")

            # Radar plot
            labels = ["Renk Tonu","Doygunluk","Parlaklık","Kontrast",
                      "Renk Canlılığı","Doyg. Sap.","Ton Sap.","Parl. Sap.","Detay"]
            θ = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
            θ = np.concatenate((θ, [θ[0]]))
            
            fig, ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(6,6))
            ax.plot(θ, np.append(A,A[0]), label="İç Dünya", lw=2)
            ax.fill(θ, np.append(A,A[0]), alpha=0.25)
            ax.plot(θ, np.append(B,B[0]), label="Dış Dünya", lw=2)
            ax.fill(θ, np.append(B,B[0]), alpha=0.25)
            ax.set_xticks(θ[:-1]) 
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_rlabel_position(30)
            ax.legend(loc="upper right", bbox_to_anchor=(1.25,1.1))
            ax.set_title("Son Görsel Metrikleri", va="bottom", pad=20)
            plt.tight_layout() 
            st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Bir hata oluştu: {str(e)}")
    st.text(traceback.format_exc())