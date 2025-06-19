# -*- coding: utf-8 -*-
"""
Streamlit app – generates highly complex abstract art based on text input
"""

# 1) Imports & env config
import os
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageChops, ImageOps, ImageFilter, ImageEnhance
import random
import hashlib
import re
import math
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

def detect_themes(text):
    """Detect themes in text using keyword matching"""
    themes = {
        "nature": ["ağaç", "orman", "çiçek", "nehir", "dağ", "gökyüzü", "deniz", "yaprak", "bitki", "doğa"],
        "urban": ["şehir", "bina", "cadde", "trafik", "köprü", "gökdelen", "kaldırım", "metro", "kent", "asfalt"],
        "cosmic": ["yıldız", "gezegen", "evren", "galaksi", "uzay", "güneş", "ay", "nebula", "kara delik", "kuasar"],
        "emotional": ["aşk", "nefret", "korku", "mutluluk", "üzüntü", "öfke", "huzur", "endişe", "coşku", "umut"],
        "organic": ["hücre", "dna", "damar", "kas", "organ", "bakteri", "mikrop", "doku", "hücresel", "biyolojik"],
        "mechanical": ["motor", "dişli", "makine", "robot", "vida", "kablo", "çark", "bıçak", "türbin", "piston"]
    }
    
    detected = []
    for theme, keywords in themes.items():
        for keyword in keywords:
            if keyword in text.lower():
                detected.append(theme)
                break  # Only need one match per theme
    
    return detected if detected else ["abstract"]

def generate_palette(text, sentiment, complexity):
    """Generate a rich color palette based on text analysis"""
    # Extract color words from text
    color_words = {
        "kırmızı": (255, 0, 0), "kirmizi": (255, 0, 0), "kızıl": (220, 20, 60),
        "mavi": (0, 0, 255), "lacivert": (0, 0, 128), "turkuaz": (64, 224, 208),
        "yeşil": (0, 255, 0), "yesil": (0, 255, 0), "zümrüt": (80, 200, 120),
        "sarı": (255, 255, 0), "sari": (255, 255, 0), "altın": (255, 215, 0),
        "mor": (128, 0, 128), "eflatun": (128, 0, 128), "lavanta": (181, 126, 220),
        "turuncu": (255, 165, 0), "amber": (255, 191, 0),
        "pembe": (255, 192, 203), "fuşya": (255, 0, 128),
        "siyah": (0, 0, 0), "beyaz": (255, 255, 255), "gri": (128, 128, 128),
        "gümüş": (192, 192, 192), "bronz": (205, 127, 50)
    }
    
    palette = []
    for word, color in color_words.items():
        if word in text.lower():
            palette.append(color)
    
    # Add theme-based colors if no specific colors mentioned
    themes = detect_themes(text)
    if not palette:
        for theme in themes:
            if theme == "nature":
                palette.extend([(34, 139, 34), (107, 142, 35), (0, 100, 0)])
            elif theme == "urban":
                palette.extend([(105, 105, 105), (169, 169, 169), (47, 79, 79)])
            elif theme == "cosmic":
                palette.extend([(25, 25, 112), (72, 61, 139), (123, 104, 238)])
            elif theme == "emotional" and sentiment > 0:
                palette.extend([(255, 69, 0), (255, 140, 0), (255, 215, 0)])
            elif theme == "emotional" and sentiment < 0:
                palette.extend([(75, 0, 130), (139, 0, 139), (148, 0, 211)])
            elif theme == "organic":
                palette.extend([(205, 92, 92), (240, 128, 128), (255, 160, 122)])
            elif theme == "mechanical":
                palette.extend([(192, 192, 192), (169, 169, 169), (128, 128, 128)])
    
    # Ensure we have enough colors
    if len(palette) < 8:
        for _ in range(8 - len(palette)):
            hue = random.random()
            saturation = 0.7 + random.random() * 0.3
            value = 0.5 + random.random() * 0.4
            r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, saturation, value)]
            palette.append((r, g, b))
    
    # Adjust colors based on sentiment
    adjusted_palette = []
    for color in palette:
        r, g, b = color
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        
        # Adjust saturation and value based on sentiment
        s = s * (1 + sentiment * 0.2)
        s = max(0.4, min(1.0, s))  # Clamp between 0.4 and 1.0
        
        v = v * (1 + sentiment * 0.1)
        v = max(0.3, min(1.0, v))  # Clamp between 0.3 and 1.0
        
        # Adjust hue based on complexity
        h = (h + complexity * 0.1) % 1.0
        
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)]
        adjusted_palette.append((r, g, b))
    
    return adjusted_palette

def generate_organic_shape(width, height, complexity=0.7):
    """Generate a complex organic shape"""
    # Create a base shape
    center_x = width // 2
    center_y = height // 2
    max_radius = min(width, height) * (0.3 + 0.3 * complexity)
    
    points = []
    num_points = int(20 + 30 * complexity)
    
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        # Add randomness to the radius
        radius_variation = 0.7 + 0.6 * random.random()
        radius = max_radius * radius_variation
        
        # Add randomness to the angle
        angle_variation = angle + (random.random() - 0.5) * math.pi / 6
        
        x = center_x + radius * math.cos(angle_variation)
        y = center_y + radius * math.sin(angle_variation)
        points.append((x, y))
    
    return points

def generate_geometric_shape(width, height, complexity=0.7):
    """Generate a complex geometric shape with recursive elements"""
    # Determine number of sides based on complexity
    num_sides = int(3 + 7 * complexity)
    
    # Create base polygon
    center_x = width // 2
    center_y = height // 2
    radius = min(width, height) * (0.3 + 0.2 * complexity)
    
    points = []
    for i in range(num_sides):
        angle = 2 * math.pi * i / num_sides
        # Add some randomness to position
        radius_variation = 0.8 + 0.4 * random.random()
        x = center_x + radius * radius_variation * math.cos(angle)
        y = center_y + radius * radius_variation * math.sin(angle)
        points.append((x, y))
    
    # Add recursive elements based on complexity
    if complexity > 0.5:
        sub_shapes = int(3 * complexity)
        for _ in range(sub_shapes):
            # Choose a point to add a sub-shape
            idx = random.randint(0, len(points)-1)
            p1 = points[idx]
            p2 = points[(idx+1) % len(points)]
            
            # Create a smaller polygon at midpoint
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            sub_radius = radius * (0.1 + 0.2 * random.random())
            sub_sides = int(3 + 4 * random.random())
            
            sub_points = []
            for j in range(sub_sides):
                angle = 2 * math.pi * j / sub_sides
                x = mid_x + sub_radius * math.cos(angle)
                y = mid_y + sub_radius * math.sin(angle)
                sub_points.append((x, y))
            
            # Add the sub-shape points
            points.extend(sub_points)
    
    return points

def generate_cosmic_shape(width, height, complexity=0.7):
    """Generate a cosmic-inspired shape with spirals and star clusters"""
    # Create a spiral base
    center_x = width // 2
    center_y = height // 2
    max_radius = min(width, height) * (0.3 + 0.2 * complexity)
    
    points = []
    num_turns = 2 + int(3 * complexity)
    num_points_per_turn = int(30 * complexity)
    
    for i in range(num_turns * num_points_per_turn):
        progress = i / (num_turns * num_points_per_turn)
        angle = 2 * math.pi * num_turns * progress
        radius = max_radius * progress
        
        # Add some randomness
        radius_var = 0.8 + 0.4 * random.random()
        angle_var = angle + (random.random() - 0.5) * math.pi / 4
        
        x = center_x + radius * radius_var * math.cos(angle_var)
        y = center_y + radius * radius_var * math.sin(angle_var)
        points.append((x, y))
    
    # Add star clusters
    num_clusters = int(5 * complexity)
    for _ in range(num_clusters):
        cluster_x = random.randint(int(width*0.2), int(width*0.8))
        cluster_y = random.randint(int(height*0.2), int(height*0.8))
        cluster_size = int(20 + 80 * complexity * random.random())
        num_stars = int(5 + 15 * complexity)
        
        for _ in range(num_stars):
            angle = random.random() * 2 * math.pi
            distance = cluster_size * random.random()
            x = cluster_x + distance * math.cos(angle)
            y = cluster_y + distance * math.sin(angle)
            points.append((x, y))
    
    return points

def generate_chaotic_shape(width, height, complexity=0.7):
    """Generate a chaotic shape with irregular patterns"""
    points = []
    num_points = int(100 + 400 * complexity)
    
    # Create a random walk
    x, y = width // 2, height // 2
    for _ in range(num_points):
        # Save current point
        points.append((x, y))
        
        # Move in a random direction
        angle = random.random() * 2 * math.pi
        distance = 1 + 10 * complexity * random.random()
        
        x += distance * math.cos(angle)
        y += distance * math.sin(angle)
        
        # Constrain to canvas
        x = max(0, min(width, x))
        y = max(0, min(height, y))
    
    return points

def generate_complex_art(text, width=1024, height=1024):
    """Generate complex abstract art based on text input with 1024px resolution"""
    # Analyze text
    clean_text = re.sub(r'[^a-zA-Z0-9ğüşıöçĞÜŞİÖÇ\s]', '', text).lower()
    words = clean_text.split()
    
    # Sentiment analysis
    positive_words = ["mutlu", "neşe", "sevgi", "huzur", "güzel", "iyi", "renkli", 
                      "aşk", "umut", "barış", "zevk", "kahkaha", "zafer", "başarı"]
    negative_words = ["korku", "endişe", "kötü", "karanlık", "üzüntü", "stres", "gri",
                      "nefret", "kaygı", "ölüm", "acı", "kayıp", "yıkım", "fırtına"]
    
    sentiment = 0
    for word in words:
        if word in positive_words: sentiment += 1
        if word in negative_words: sentiment -= 1
    
    # Calculate complexity
    complexity = min(1.0, max(0.3, len(set(words)) / 15))
    
    # Generate palette
    palette = generate_palette(text, sentiment, complexity)
    
    # Create hash for consistent results
    hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    random.seed(hash_val)
    
    # Create blank image
    img = Image.new('RGB', (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Generate gradient background
    bg_color1 = random.choice(palette)
    bg_color2 = random.choice([c for c in palette if c != bg_color1])
    
    for y in range(height):
        # Calculate gradient
        ratio = y / height
        r = int(bg_color1[0] * (1 - ratio) + bg_color2[0] * ratio)
        g = int(bg_color1[1] * (1 - ratio) + bg_color2[1] * ratio)
        b = int(bg_color1[2] * (1 - ratio) + bg_color2[2] * ratio)
        
        # Draw gradient line
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # Detect themes to determine shape style
    themes = detect_themes(text)
    shape_generators = []
    
    if "nature" in themes:
        shape_generators.append(generate_organic_shape)
    if "urban" in themes:
        shape_generators.append(generate_geometric_shape)
    if "cosmic" in themes:
        shape_generators.append(generate_cosmic_shape)
    if "emotional" in themes or "chaotic" in themes:
        shape_generators.append(generate_chaotic_shape)
    
    # Fallback if no specific themes detected
    if not shape_generators:
        shape_generators = [generate_organic_shape, generate_geometric_shape, 
                            generate_cosmic_shape, generate_chaotic_shape]
    
    # Generate multiple shape layers
    num_layers = int(3 + 5 * complexity)
    for layer in range(num_layers):
        # Create a new transparent layer
        layer_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer_img)
        
        # Choose shape generator for this layer
        shape_gen = random.choice(shape_generators)
        
        # Generate multiple shapes in this layer
        num_shapes = int(1 + 4 * complexity)
        for _ in range(num_shapes):
            # Random position and size
            center_x = random.randint(int(width*0.1), int(width*0.9))
            center_y = random.randint(int(height*0.1), int(height*0.9))
            shape_width = int(width * (0.1 + 0.4 * random.random()))
            shape_height = int(height * (0.1 + 0.4 * random.random()))
            
            # Generate shape points
            points = shape_gen(shape_width, shape_height, complexity)
            
            # Move shape to position
            min_x = min(p[0] for p in points)
            min_y = min(p[1] for p in points)
            max_x = max(p[0] for p in points)
            max_y = max(p[1] for p in points)
            
            scale_x = shape_width / (max_x - min_x) if max_x != min_x else 1
            scale_y = shape_height / (max_y - min_y) if max_y != min_y else 1
            
            scaled_points = []
            for x, y in points:
                scaled_x = center_x + (x - min_x) * scale_x
                scaled_y = center_y + (y - min_y) * scale_y
                scaled_points.append((scaled_x, scaled_y))
            
            # Choose color with transparency
            color = random.choice(palette)
            alpha = int(30 + 200 * random.random() * complexity)
            fill_color = color + (alpha,)
            
            # Draw the shape
            layer_draw.polygon(scaled_points, fill=fill_color)
        
        # Apply transformations to the layer
        if random.random() > 0.5:
            # Rotate without expanding to maintain size
            layer_img = layer_img.rotate(
                random.randint(-30, 30), 
                resample=Image.BICUBIC, 
                expand=False  # CRITICAL FIX: Maintain original size
            )
        
        # Blend layer into main image
        img = Image.alpha_composite(img.convert('RGBA'), layer_img)
    
    # Convert back to RGB
    img = img.convert('RGB')
    
    # Add texture overlay
    texture = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    texture_draw = ImageDraw.Draw(texture)
    
    num_texture_points = int(5000 * complexity)
    for _ in range(num_texture_points):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(1, 5)
        color = random.choice(palette)
        alpha = random.randint(10, 50)
        texture_draw.ellipse([x, y, x+size, y+size], fill=color + (alpha,))
    
    img = Image.alpha_composite(img.convert('RGBA'), texture).convert('RGB')
    
    # Apply final enhancements
    if sentiment > 1:
        # Bright and vibrant
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.3)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
    elif sentiment < -1:
        # Dark and moody
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.8)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Add sharpening for detail
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)
    
    return img

# 4) UI
st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")
st.info("ℹ️ Metin girişlerinize göre otomatik olarak oluşturulan soyut sanat eserleri")

# Custom CSS for better layout
st.markdown("""
<style>
    .text-box {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 20px;
        background-color: #f9f9f9;
    }
    .art-title {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .art-analysis {
        font-size: 0.9em;
        color: #555;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="text-box">', unsafe_allow_html=True)
    inner_txt = st.text_area("📖 İç Dünya (Duygu, düşünce veya rüyalarınızı birkaç cümle ile anlatın):", height=120, 
                            value="")
    st.markdown('</div>', unsafe_allow_html=True)
    
with col2:
    st.markdown('<div class="text-box">', unsafe_allow_html=True)
    outer_txt = st.text_area("🌍 Dış Dünya (Çevrenizdekileri, duyduğunuz, gördüğünüz, dokunduğunuz vs. şeyleri birkaç cümle ile anlatın):", height=120, 
                            value="")
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🎨 Oluştur ve Karşılaştır", use_container_width=True):
    if not inner_txt or not outer_txt:
        st.warning("⚠️ Lütfen her iki metni de girin.")
        st.stop()

    with st.spinner("🖼️ Yüksek çözünürlüklü soyut sanat eserleri oluşturuluyor..."):
        img1 = generate_complex_art(inner_txt)
        img2 = generate_complex_art(outer_txt)

    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="art-title">İç Dünya</div>', unsafe_allow_html=True)
        st.image(img1, use_container_width=True)
        
        # Text analysis
        themes = detect_themes(inner_txt)
        st.markdown(f'<div class="art-analysis"><b>Analiz:</b> {", ".join(themes)} temaları | {len(inner_txt.split())} kelime</div>', 
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="art-title">Dış Dünya</div>', unsafe_allow_html=True)
        st.image(img2, use_container_width=True)
        
        # Text analysis
        themes = detect_themes(outer_txt)
        st.markdown(f'<div class="art-analysis"><b>Analiz:</b> {", ".join(themes)} temaları | {len(outer_txt.split())} kelime</div>', 
                   unsafe_allow_html=True)

    # Compute metrics
    try:
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        m1 = compute_metrics(arr1)
        m2 = compute_metrics(arr2)
        iou = calculate_iou(m1, m2)
        
        # Create styled similarity display
        similarity_color = "#4CAF50" if iou > 0.5 else "#FF9800" if iou > 0.3 else "#F44336"
        st.markdown(f"""
        <div style="background-color: #f0f0f0; border-radius: 5px; padding: 15px; margin: 20px 0; text-align: center;">
            <h3 style="color: #333; margin-bottom: 10px;">🔍 Görsel Benzerlik Analizi</h3>
            <div style="font-size: 2em; font-weight: bold; color: {similarity_color};">{iou:.3f}</div>
            <div style="margin-top: 10px; color: #666;">0 (tamamen farklı) - 1 (tamamen benzer)</div>
        </div>
        """, unsafe_allow_html=True)

        # Radar chart
        labels = ["Parlaklık", "Kontrast", "Renk Canlılığı", "Detay"]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        m1_plot = np.concatenate([m1, [m1[0]]])
        m2_plot = np.concatenate([m2, [m2[0]]])

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        ax.plot(angles, m1_plot, 'o-', linewidth=2, label='İç Dünya', color='#4CAF50')
        ax.fill(angles, m1_plot, alpha=0.25, color='#4CAF50')
        ax.plot(angles, m2_plot, 'o-', linewidth=2, label='Dış Dünya', color='#2196F3')
        ax.fill(angles, m2_plot, alpha=0.25, color='#2196F3')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_title("Görsel Metrik Karşılaştırması", fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set radial grid
        ax.set_rlabel_position(180)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"❌ Analiz hatası: {str(e)}")