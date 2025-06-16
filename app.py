# -*- coding: utf-8 -*-
"""
Streamlit app – compares abstract "inner" & "outer" world images.
Optimized for Streamlit Cloud with minimal resource usage
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
import sys
import traceback

# Configure environment for Streamlit Cloud
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

import streamlit as st
st.set_page_config(page_title="Soyut İç & Dış Dünya", layout="wide")

# Health check handling
if os.environ.get("ST_STATE") == "health-check":
    st.stop()

# Get Hugging Face token
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)

if not HF_TOKEN:
    st.error("❌ Hugging Face token eksik! Lütfen Streamlit Secrets'a ekleyin.")
    st.stop()

try:
    # Lightweight imports only
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    # Cache the pipeline model
    @st.cache_resource(show_spinner=False)
    def load_pipeline():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32  # Use float32 for better compatibility
        
        # Use a smaller model to reduce memory requirements
        model_name = "prompthero/openjourney"
        
        # HF_TOKEN is read from Streamlit secrets automatically
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        # switch to a faster scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        if device == "cpu":
            pipe.enable_attention_slicing()
        
        return pipe.to(device)

    def compute_metrics(arr):
        """Compute image metrics without complex operations"""
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        brightness = (r + g + b).mean() / 3
        contrast = np.std(r) + np.std(g) + np.std(b)
        rg = r - g
        yb = 0.5 * (r + g) - b
        colorfulness = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gx, gy = np.gradient(gray)
        detail = np.mean(np.sqrt(gx**2 + gy**2))
        return [brightness, contrast, colorfulness, detail]

    def calculate_iou(A, B):
        """Simplified similarity calculation"""
        A_norm = (A - A.min()) / (A.max() - A.min() + 1e-10)
        B_norm = (B - B.min()) / (B.max() - B.min() + 1e-10)
        intersection = np.sum(np.minimum(A_norm, B_norm))
        union = np.sum(np.maximum(A_norm, B_norm))
        return intersection / union if union > 0 else 0

    # Streamlit UI
    st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")

    inner_txt = st.text_area("📖 İç Dünya:", height=120,
                            value="Duygularım, düşüncelerim, içsel yolculuğum")
    outer_txt = st.text_area("🌍 Dış Dünya:", height=120,
                            value="Sosyal ilişkiler, doğa, şehir yaşamı")

    if st.button("🎨 Oluştur ve Karşılaştır"):
        if not inner_txt or not outer_txt:
            st.warning("⚠️ Lütfen her iki metni de girin.")
        else:
            with st.spinner("📥 Model yükleniyor… (Bu ilk seferde 1-2 dakika sürebilir)"):
                pipe = load_pipeline()
            
            with st.spinner("🖼️ Görseller üretiliyor… (Lütfen bekleyiniz)"):
                img1 = pipe(
                    f"mdjrny-v4 style abstract art: {inner_txt}",
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=128,
                    width=128
                ).images[0]
                
                img2 = pipe(
                    f"mdjrny-v4 style abstract art: {outer_txt}",
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=128,
                    width=128
                ).images[0]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("İç Dünya")
                st.image(img1, use_column_width=True)
            with col2:
                st.subheader("Dış Dünya")
                st.image(img2, use_column_width=True)

            arr1 = np.array(img1)
            arr2 = np.array(img2)
            metrics1 = compute_metrics(arr1)
            metrics2 = compute_metrics(arr2)
            
            iou = calculate_iou(np.array(metrics1), np.array(metrics2))
            st.markdown(f"**🔍 Benzerlik Oranı: {iou:.3f}**")
            
            labels = ["Parlaklık", "Kontrast", "Renk Canlılığı", "Detay"]
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"polar": True})
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]
            
            metrics1 += metrics1[:1]
            metrics2 += metrics2[:1]
            
            ax.plot(angles, metrics1, "o-", label="İç Dünya")
            ax.fill(angles, metrics1, alpha=0.25)
            ax.plot(angles, metrics2, "o-", label="Dış Dünya")
            ax.fill(angles, metrics2, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_title("Görsel Metrik Karşılaştırması")
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            
            st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Bir hata oluştu: {str(e)}")
    st.text(traceback.format_exc())