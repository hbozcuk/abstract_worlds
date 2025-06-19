# -*- coding: utf-8 -*-
"""
Streamlit app – generates abstract art based on inner/outer world concepts
"""

# 1) Imports & env config
import os
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import random
import hashlib
import re

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

def text_to_art_style(text):
    """Convert text to art style parameters"""
    # Clean text
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    
    # Create hash for consistent results
    hash_val = int(hashlib.sha256(clean_text.encode()).hexdigest(), 16)
    random.seed(hash_val)
    
    # Determine art style based on text sentiment
    positive_words = ["mutlu", "neşe", "sevgi", "huzur", "güzel", "iyi", "renkli"]
    negative_words = ["korku", "endişe", "kötü", "karanlık", "üzüntü", "stres", "gri"]
    
    sentiment = 0
    for word in clean_text.split():
        if word in positive_words: sentiment += 1
        if word in negative_words: sentiment -= 1
    
    # Set parameters based on sentiment
    if sentiment > 0:
        return {
            "color_range": ((50, 200), (50, 200), (50, 200)),  # Bright colors
            "shape_count": random.randint(15, 30),
            "blur_radius": random.uniform(0.5, 2),
            "complexity": "high"
        }
    elif sentiment < 0:
        return {
            "color_range": ((0, 100), (0, 100), (0, 100)),  # Dark colors
            "shape_count": random.randint(5, 15),
            "blur_radius": random.uniform(2, 5),
            "complexity": "low"
        }
    else:
        return {
            "color_range": ((0, 200), (0, 200), (0, 200)),  # Mixed colors
            "shape_count": random.randint(10, 25),
            "blur_radius": random.uniform(1, 3),
            "complexity": "medium"
        }

def generate_abstract_art(text, width=512, height=512):
    """Generate abstract art based on text input"""
    # Get style parameters from text
    style = text_to_art_style(text)
    
    # Create blank image
    img = Image.new('RGB', (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Generate shapes based on text
    for _ in range(style["shape_count"]):
        # Get random color based on style
        r_range, g_range, b_range = style["color_range"]
        color = (
            random.randint(*r_range),
            random.randint(*g_range),
            random.randint(*b_range)
        )
        
        # Choose shape type
        shape_type = random.choice(["circle", "rectangle", "polygon"])
        
        # Generate position and size
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        size = random.randint(20, min(width, height)//3)
        
        if shape_type == "circle":
            draw.ellipse([x1, y1, x1+size, y1+size], fill=color, outline=None)
        elif shape_type == "rectangle":
            rotation = random.randint(0, 45)
            # Create rotated rectangle
            rect_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            rect_draw = ImageDraw.Draw(rect_img)
            rect_draw.rectangle([0, 0, size, size], fill=color)
            rect_img = rect_img.rotate(rotation, expand=True)
            img.paste(rect_img, (x1, y1), rect_img)
        else:  # polygon
            points = []
            for _ in range(random.randint(3, 6)):
                points.append((
                    x1 + random.randint(-size//2, size//2),
                    y1 + random.randint(-size//2, size//2)
                ))
            draw.polygon(points, fill=color, outline=None)
    
    # Add texture based on complexity
    if style["complexity"] == "high":
        for _ in range(50):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            size = random.randint(2, 10)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.ellipse([x1, y1, x1+size, y1+size], fill=color)
    
    # Apply blur for artistic effect
    img = img.filter(ImageFilter.GaussianBlur(radius=style["blur_radius"]))
    
    return img

# 4) UI
st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")
st.info("ℹ️ Metin girişlerinize göre otomatik olarak oluşturulan soyut sanat eserleri")

inner_txt = st.text_area("📖 İç Dünya:", height=120, value="Rüyalarımda gördüğüm renkli dünya")
outer_txt = st.text_area("🌍 Dış Dünya:", height=120, value="Şehirdeki gri binalar ve trafik")

if st.button("🎨 Oluştur ve Karşılaştır"):
    if not inner_txt or not outer_txt:
        st.warning("⚠️ Lütfen her iki metni de girin.")
        st.stop()

    with st.spinner("🖼️ Soyut sanat eserleri oluşturuluyor..."):
        img1 = generate_abstract_art(inner_txt)
        img2 = generate_abstract_art(outer_txt)

    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("İç Dünya")
        st.image(img1, use_container_width=True)
    with col2:
        st.subheader("Dış Dünya")
        st.image(img2, use_container_width=True)

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