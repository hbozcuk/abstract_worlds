# -*- coding: utf-8 -*-
"""
Streamlit app – compares abstract “inner” & “outer” world images.
No model files download during Streamlit-Cloud health-check.
"""

##############################################################################
# 0)  Minimal Streamlit setup
##############################################################################
import os, sys, types
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"

import streamlit as st
st.set_page_config(page_title="Soyut İç & Dış Dünya", layout="wide")

##############################################################################
# 1)  Abort immediately during Cloud health-probe
##############################################################################
if os.environ.get("ST_STATE") == "health-check":
    st.stop()

##############################################################################
# 2)  Safe stub for diffusers
##############################################################################
class _DiffusersStub(types.ModuleType):
    _is_stub = True
    _SAFE = {
        "__name__", "__doc__", "__package__", "__loader__", "__spec__",
        "__file__", "__path__", "__dict__"
    }

    # preset normal module attrs so hasattr() works
    __file__ = "<stub>"
    __path__ = []          # makes it a package
    __spec__ = None
    __loader__ = None

    def __getattr__(self, name):
        if name in self._SAFE:
            return None
        raise RuntimeError("🚫 diffusers is stubbed until the button is pressed.")

# register stub only if diffusers not already imported
sys.modules.setdefault("diffusers", _DiffusersStub("diffusers"))

##############################################################################
# 3)  Lightweight imports
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon

##############################################################################
# 4)  Lazy loader – swaps stub for real diffusers on demand
##############################################################################
_PIPE = None

def load_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    # Remove stub → import real library
    if getattr(sys.modules.get("diffusers"), "_is_stub", False):
        del sys.modules["diffusers"]

    import torch
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    _PIPE = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=dtype
    ).to(device)

    if device == "cpu":
        _PIPE.enable_attention_slicing()

    return _PIPE

##############################################################################
# 5)  Metric helper
##############################################################################
def compute_metrics(path: str):
    img  = Image.open(path).convert("RGB")
    arr  = np.array(img)
    hsv  = img.convert("HSV")
    h,s,v = np.array(hsv).transpose(2,0,1)
    hue_mean, sat_mean, val_mean = h.mean(), s.mean(), v.mean()
    sat_std, hue_std, val_std    = s.std(), h.std(), v.std()
    gray = np.array(img.convert("L"), float)
    contrast = gray.std()
    R,G,B = arr[...,0].astype(float), arr[...,1].astype(float), arr[...,2].astype(float)
    rg, yb = R-G, 0.5*(R+G)-B
    colorfulness = np.sqrt(rg.std()**2 + yb.std()**2) + 0.3*np.sqrt(rg.mean()**2 + yb.mean()**2)
    gy,gx = np.gradient(gray)
    detail = np.mean(np.sqrt(gx**2 + gy**2))
    return [hue_mean,sat_mean,val_mean,contrast,colorfulness,
            sat_std,hue_std,val_std,detail]

##############################################################################
# 6)  Streamlit UI
##############################################################################
st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")

inner = st.text_area("📖 İç Dünya:",  height=120)
outer = st.text_area("🌍 Dış Dünya:", height=120)

if st.button("🎨 Oluştur ve Karşılaştır"):
    if not inner or not outer:
        st.warning("⚠️ Lütfen her iki metni de girin.")
        st.stop()

    with st.spinner("📥 Model yükleniyor…"):
        pipe = load_pipeline()

    with st.spinner("🖼️ Görseller üretiliyor…"):
        p1 = f"Soyut sanat eseri, non-figurative geometric abstraction: {inner}"
        p2 = f"Soyut sanat eseri, non-figurative geometric abstraction: {outer}"
        img1 = pipe(p1, num_inference_steps=50, guidance_scale=9.0).images[0]
        img2 = pipe(p2, num_inference_steps=50, guidance_scale=9.0).images[0]

    col1, col2 = st.columns(2)
    col1.subheader("İç Dünya");  col1.image(img1, use_column_width=True)
    col2.subheader("Dış Dünya"); col2.image(img2, use_column_width=True)

    img1.save("inner_tmp.png"); img2.save("outer_tmp.png")
    A = np.array(compute_metrics("inner_tmp.png"))
    B = np.array(compute_metrics("outer_tmp.png"))

    labels = ["Renk Tonu","Doygunluk","Parlaklık","Kontrast",
              "Renk Canlılığı","Doyg. Sap.","Ton Sap.","Parl. Sap.","Detay"]
    θ = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    θ = np.concatenate((θ,[θ[0]]))
    ptsA = [(r*np.cos(a), r*np.sin(a)) for r,a in zip(np.append(A,A[0]), θ)]
    ptsB = [(r*np.cos(a), r*np.sin(a)) for r,a in zip(np.append(B,B[0]), θ)]
    iou = Polygon(ptsA).intersection(Polygon(ptsB)).area / Polygon(ptsA).union(Polygon(ptsB)).area
    st.markdown(f"**🔍 Benzerlik (IoU): {iou:.3f}**")

    fig, ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(6,6))
    ax.plot(θ, np.append(A,A[0]), label="İç Dünya", lw=2)
    ax.fill(θ, np.append(A,A[0]), alpha=0.25)
    ax.plot(θ, np.append(B,B[0]), label="Dış Dünya", lw=2)
    ax.fill(θ, np.append(B,B[0]), alpha=0.25)
    ax.set_xticks(θ[:-1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_rlabel_position(30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25,1.1))
    ax.set_title("Son Görsel Metrikleri", va="bottom", pad=20)
    plt.tight_layout(); st.pyplot(fig)