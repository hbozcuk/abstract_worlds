# -*- coding: utf-8 -*-
"""
Streamlit app – generates complex abstract art based on inner/outer world concepts
"""

# 1) Imports & env config
import os
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageChops, ImageOps, ImageFilter
import random
import hashlib
import re
import math
from collections import Counter
import colorsys

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

def analyze_text(text):
    """Analyze text to extract key features for art generation"""
    # Clean and tokenize text
    clean_text = re.sub(r'[^a-zA-Z0-9ğüşıöçĞÜŞİÖÇ\s]', '', text).lower()
    words = clean_text.split()
    
    # Sentiment analysis
    positive_words = ["mutlu", "neşe", "sevgi", "huzur", "güzel", "iyi", "renkli", 
                      "aşk", "umut", "barış", "zevk", "kahkaha", "zafer", "zafer"]
    negative_words = ["korku", "endişe", "kötü", "karanlık", "üzüntü", "stres", "gri",
                      "nefret", "kaygı", "ölüm", "acı", "kayıp", "yıkım", "fırtına"]
    
    sentiment = 0
    for word in words:
        if word in positive_words: sentiment += 1
        if word in negative_words: sentiment -= 1
    
    # Color analysis
    color_words = {
        "kırmızı": (255, 0, 0), "kirmizi": (255, 0, 0),
        "mavi": (0, 0, 255), "yeşil": (0, 255, 0), "yesil": (0, 255, 0),
        "sarı": (255, 255, 0), "sari": (255, 255, 0), "mor": (128, 0, 128),
        "turuncu": (255, 165, 0), "pembe": (255, 192, 203), "siyah": (0, 0, 0),
        "beyaz": (255, 255, 255), "gri": (128, 128, 128), "altın": (255, 215, 0),
        "gümüş": (192, 192, 192), "mavi": (0, 0, 255), "lacivert": (0, 0, 128),
        "turkuaz": (64, 224, 208), "eflatun": (128, 0, 128), "bej": (245, 245, 220)
    }
    
    colors_in_text = []
    for word in words:
        if word in color_words:
            colors_in_text.append(color_words[word])
    
    # Energy level (based on word count and sentiment)
    energy = min(1.0, max(0.1, len(words) / 50))
    
    # Complexity (based on unique words)
    complexity = min(1.0, max(0.3, len(set(words)) / 20))
    
    return {
        "sentiment": sentiment,
        "colors": colors_in_text,
        "energy": energy,
        "complexity": complexity,
        "word_count": len(words),
        "unique_words": len(set(words))
    }

def generate_palette(colors, sentiment, complexity):
    """Generate a color palette based on text analysis"""
    # Base palette from text colors
    palette = colors if colors else []
    
    # Add colors based on sentiment
    if sentiment > 1:
        palette.extend([(255, 215, 0), (255, 165, 0), (255, 69, 0)])  # Warm colors
    elif sentiment < -1:
        palette.extend([(75, 0, 130), (0, 0, 128), (25, 25, 112)])  # Cool colors
    else:
        palette.extend([(50, 205, 50), (30, 144, 255), (138, 43, 226)])  # Balanced
    
    # Add complexity-based colors
    if complexity > 0.7:
        palette.extend([(220, 20, 60), (0, 255, 127), (148, 0, 211)])
    
    # Ensure minimum palette size
    if len(palette) < 6:
        palette.extend([(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                        for _ in range(6 - len(palette))])
    
    # Adjust colors based on sentiment
    adjusted_palette = []
    for color in palette:
        r, g, b = color
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        
        # Adjust lightness based on sentiment
        if sentiment < -2:
            l = max(0.1, l * 0.7)
        elif sentiment > 2:
            l = min(0.9, l * 1.3)
        
        # Adjust saturation based on complexity
        s = min(0.9, s * (1 + complexity * 0.5))
        
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        adjusted_palette.append((int(r*255), int(g*255), int(b*255)))
    
    return adjusted_palette

def generate_complex_art(text, width=512, height=512):
    """Generate complex abstract art based on text input"""
    # Analyze text
    analysis = analyze_text(text)
    palette = generate_palette(analysis["colors"], analysis["sentiment"], analysis["complexity"])
    
    # Create hash for consistent results
    hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    random.seed(hash_val)
    
    # Create blank image
    img = Image.new('RGB', (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Generate Perlin-like noise background
    noise_layer = Image.new('RGB', (width, height))
    noise_draw = ImageDraw.Draw(noise_layer)
    for _ in range(int(analysis["complexity"] * 5000)):
        x, y = random.randint(0, width), random.randint(0, height)
        size = random.randint(1, 5)
        color = random.choice(palette)
        noise_draw.ellipse([x, y, x+size, y+size], fill=color)
    img = ImageChops.add(img, noise_layer)
    
    # Generate organic shapes
    for _ in range(int(analysis["complexity"] * 20 + 5)):
        # Create a new layer for each shape
        shape_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        shape_draw = ImageDraw.Draw(shape_layer)
        
        # Choose random parameters
        color = random.choice(palette)
        alpha = random.randint(50, 200)
        shape_type = random.choice(["blob", "wave", "fractal"])
        
        if shape_type == "blob":
            # Generate organic blob
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)
            radius = random.randint(20, min(width, height) // 4)
            points = []
            for i in range(12):
                angle = i * (2 * math.pi / 12)
                variation = random.uniform(0.7, 1.3)
                px = center_x + radius * variation * math.cos(angle)
                py = center_y + radius * variation * math.sin(angle)
                points.append((px, py))
            shape_draw.polygon(points, fill=(color[0], color[1], color[2], alpha))
        
        elif shape_type == "wave":
            # Generate wave-like pattern
            num_points = random.randint(4, 8)
            points = []
            for i in range(num_points):
                x = width * i / (num_points - 1)
                base_y = random.randint(0, height)
                amplitude = random.randint(20, 100)
                y = base_y + amplitude * math.sin(i * 1.5)
                points.append((x, y))
            
            # Add bottom points to close shape
            points.append((width, height))
            points.append((0, height))
            
            shape_draw.polygon(points, fill=(color[0], color[1], color[2], alpha))
        
        else:  # fractal
            # Generate fractal pattern
            def draw_fractal(draw, x, y, size, depth, max_depth):
                if depth > max_depth:
                    return
                
                # Draw current level
                draw.ellipse([x-size, y-size, x+size, y+size], 
                            outline=(color[0], color[1], color[2], alpha), 
                            width=random.randint(1, 3))
                
                # Recursive calls
                for i in range(random.randint(3, 6)):
                    angle = random.uniform(0, 2 * math.pi)
                    distance = size * random.uniform(0.5, 1.5)
                    new_x = x + distance * math.cos(angle)
                    new_y = y + distance * math.sin(angle)
                    new_size = size * random.uniform(0.3, 0.7)
                    draw_fractal(draw, new_x, new_y, new_size, depth+1, max_depth)
            
            max_depth = min(5, int(analysis["complexity"] * 4) + 1)
            start_x = random.randint(0, width)
            start_y = random.randint(0, height)
            start_size = random.randint(20, 80)
            draw_fractal(shape_draw, start_x, start_y, start_size, 0, max_depth)
        
        # Blend shape into main image
        img = Image.alpha_composite(img.convert('RGBA'), shape_layer)
    
    # Convert back to RGB
    img = img.convert('RGB')
    
    # Apply texture overlay
    texture = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    texture_draw = ImageDraw.Draw(texture)
    for _ in range(int(analysis["complexity"] * 1000)):
        x, y = random.randint(0, width), random.randint(0, height)
        size = random.randint(1, 3)
        alpha = random.randint(10, 50)
        color = random.choice(palette)
        texture_draw.ellipse([x, y, x+size, y+size], 
                            fill=(color[0], color[1], color[2], alpha))
    img = Image.alpha_composite(img.convert('RGBA'), texture).convert('RGB')
    
    # Apply final filters based on sentiment
    if analysis["sentiment"] < -2:
        # Dark and moody
        img = ImageOps.autocontrast(img, cutoff=2)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.8)
    elif analysis["sentiment"] > 2:
        # Bright and vibrant
        img = ImageOps.autocontrast(img, cutoff=0.5)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.3)
    else:
        # Balanced
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.SMOOTH_MORE)
    
    return img

# 4) UI
st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")
st.info("ℹ️ Metin girişlerinize göre otomatik olarak oluşturulan karmaşık soyut sanat eserleri")

inner_txt = st.text_area("📖 İç Dünya:", height=120, value="Rüyalarımda gördüğüm renkli dünya, sonsuz olasılıklar ve neşeli kaos")
outer_txt = st.text_area("🌍 Dış Dünya:", height=120, value="Şehirdeki gri binalar, trafik karmaşası ve sistematik düzen")

if st.button("🎨 Oluştur ve Karşılaştır"):
    if not inner_txt or not outer_txt:
        st.warning("⚠️ Lütfen her iki metni de girin.")
        st.stop()

    with st.spinner("🖼️ Karmaşık soyut sanat eserleri oluşturuluyor..."):
        img1 = generate_complex_art(inner_txt)
        img2 = generate_complex_art(outer_txt)

    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("İç Dünya")
        st.image(img1, use_container_width=True)
        st.caption(f"Metin analizi: {inner_txt}")
    with col2:
        st.subheader("Dış Dünya")
        st.image(img2, use_container_width=True)
        st.caption(f"Metin analizi: {outer_txt}")

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