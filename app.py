# -*- coding: utf-8 -*-
"""
Streamlit app – generates highly complex abstract art based on text input with AI analysis
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
import requests
import json
import time

# 2) Page config
st.set_page_config(page_title="Soyut İç & Dış Dünya", layout="wide")

# Hugging Face API token (Streamlit secrets'ten alınacak)
HF_TOKEN = st.secrets.get("HF_TOKEN", "your_huggingface_token_here")
API_URL = "https://api-inference.huggingface.co/models"

# 3) Utility functions
def compute_metrics(arr: np.ndarray):
    """
    Compute image metrics:
    1. Brightness (Parlaklık)
    2. Contrast (Kontrast)
    3. Colorfulness (Renk Canlılığı)
    4. Complexity (Karmaşıklık)
    """
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
    
    # 1. Brightness
    brightness = (r + g + b).mean() / 3
    
    # 2. Contrast
    contrast = np.std(r) + np.std(g) + np.std(b)
    
    # 3. Colorfulness
    rg, yb = r - g, 0.5*(r+g) - b
    colorfulness = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    
    # 4. Complexity - measure of visual complexity
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gx, gy = np.gradient(gray)
    edge_strength = np.sqrt(gx**2 + gy**2)
    complexity = np.mean(edge_strength) + np.std(edge_strength)
    
    return [brightness, contrast, colorfulness, complexity]

def calculate_iou(A, B):
    A, B = np.array(A), np.array(B)
    A_norm = (A - A.min()) / (A.max() - A.min() + 1e-10)
    B_norm = (B - B.min()) / (B.max() - B.min() + 1e-10)
    return np.sum(np.minimum(A_norm, B_norm)) / np.sum(np.maximum(A_norm, B_norm))

def ai_text_analysis(text):
    """AI-powered text analysis using Hugging Face models"""
    try:
        # Duygu analizi
        sentiment_payload = {"inputs": text}
        sentiment_response = requests.post(
            f"{API_URL}/distilbert-base-uncased-finetuned-sst-2-english",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json=sentiment_payload
        )
        sentiment_result = sentiment_response.json()[0]
        
        # Temaları analiz et
        theme_payload = {
            "inputs": text,
            "parameters": {
                "candidate_labels": ["doğa", "şehir", "uzay", "duygusal", "organik", "mekanik", "soyut", "kaotik"]
            }
        }
        theme_response = requests.post(
            f"{API_URL}/typeform/distilbert-base-uncased-mnli",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json=theme_payload
        )
        theme_result = theme_response.json()
        
        # Renk analizi
        color_payload = {"inputs": text}
        color_response = requests.post(
            f"{API_URL}/sentence-transformers/all-MiniLM-L6-v2",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json=color_payload
        )
        color_embeddings = color_response.json()
        
        # Sonuçları birleştir
        return {
            "sentiment": sentiment_result,
            "themes": theme_result,
            "color_embeddings": color_embeddings
        }
    
    except Exception as e:
        # Hata durumunda basit bir analiz döndür
        positive_words = ["mutlu", "neşe", "sevgi", "huzur", "güzel", "iyi", "renkli"]
        negative_words = ["korku", "endişe", "kötü", "karanlık", "üzüntü", "stres", "gri"]
        
        sentiment_score = 0
        words = text.lower().split()
        for word in words:
            if word in positive_words: sentiment_score += 1
            if word in negative_words: sentiment_score -= 1
        
        return {
            "sentiment": [{"label": "POSITIVE" if sentiment_score > 0 else "NEGATIVE", "score": abs(sentiment_score)/10}],
            "themes": {"labels": ["soyut"], "scores": [1.0]},
            "color_embeddings": []
        }

def generate_palette(text, ai_analysis):
    """Generate a rich color palette based on AI text analysis"""
    # Renk kelimeleri sözlüğü
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
    
    # Metinde geçen renk kelimelerini bul
    palette = []
    text_lower = text.lower()
    for word, color in color_words.items():
        if word in text_lower:
            palette.append(color)
    
    # AI renk analizini kullan
    if ai_analysis["color_embeddings"]:
        try:
            # Renk vektörlerinin ortalamasını al
            color_embeddings = np.array(ai_analysis["color_embeddings"])
            avg_embedding = np.mean(color_embeddings, axis=0)
            
            # Vektörden renk değerleri oluştur
            r = int((avg_embedding[0] + 1) * 127.5) % 256
            g = int((avg_embedding[1] + 1) * 127.5) % 256
            b = int((avg_embedding[2] + 1) * 127.5) % 256
            palette.append((r, g, b))
        except:
            pass
    
    # Yeterli renk yoksa rastgele ekle
    if len(palette) < 8:
        for _ in range(8 - len(palette)):
            hue = random.random()
            saturation = 0.7 + random.random() * 0.3
            value = 0.5 + random.random() * 0.4
            r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, saturation, value)]
            palette.append((r, g, b))
    
    return palette

def generate_organic_shape(width, height, complexity=0.7):
    """Generate a complex organic shape with higher complexity"""
    center_x = width // 2
    center_y = height // 2
    max_radius = min(width, height) * (0.3 + 0.3 * complexity)
    
    points = []
    num_points = int(30 + 50 * complexity)
    
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        radius_variation = 0.5 + 0.8 * random.random()
        radius = max_radius * radius_variation
        angle_variation = angle + (random.random() - 0.5) * math.pi / 4 * (1 + complexity)
        x = center_x + radius * math.cos(angle_variation)
        y = center_y + radius * math.sin(angle_variation)
        points.append((x, y))
    
    if complexity > 0.5:
        sub_shapes = int(3 + 5 * complexity)
        for _ in range(sub_shapes):
            idx = random.randint(0, len(points)-1)
            p1 = points[idx]
            p2 = points[(idx+1) % len(points)]
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            sub_radius = max_radius * (0.15 + 0.25 * random.random())
            num_sub_points = int(10 + 20 * complexity)
            sub_points = []
            for j in range(num_sub_points):
                sub_angle = 2 * math.pi * j / num_sub_points
                sub_radius_var = 0.7 + 0.6 * random.random()
                x = mid_x + sub_radius * sub_radius_var * math.cos(sub_angle)
                y = mid_y + sub_radius * sub_radius_var * math.sin(sub_angle)
                sub_points.append((x, y))
            points.extend(sub_points)
    
    return points

def generate_geometric_shape(width, height, complexity=0.7):
    """Generate a complex geometric shape with more elements"""
    num_sides = int(5 + 10 * complexity)
    center_x = width // 2
    center_y = height // 2
    radius = min(width, height) * (0.3 + 0.2 * complexity)
    
    points = []
    for i in range(num_sides):
        angle = 2 * math.pi * i / num_sides
        radius_variation = 0.7 + 0.6 * random.random()
        angle_variation = angle + (random.random() - 0.5) * math.pi / 8
        x = center_x + radius * radius_variation * math.cos(angle_variation)
        y = center_y + radius * radius_variation * math.sin(angle_variation)
        points.append((x, y))
    
    if complexity > 0.4:
        sub_shapes = int(4 + 6 * complexity)
        for _ in range(sub_shapes):
            idx = random.randint(0, len(points)-1)
            p1 = points[idx]
            p2 = points[(idx+1) % len(points)]
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            sub_radius = radius * (0.1 + 0.25 * random.random())
            sub_sides = int(3 + 7 * random.random())
            sub_points = []
            for j in range(sub_sides):
                sub_angle = 2 * math.pi * j / sub_sides
                sub_radius_var = 0.8 + 0.4 * random.random()
                x = mid_x + sub_radius * sub_radius_var * math.cos(sub_angle)
                y = mid_y + sub_radius * sub_radius_var * math.sin(sub_angle)
                sub_points.append((x, y))
            points.extend(sub_points)
    
    return points

def generate_cosmic_shape(width, height, complexity=0.7):
    """Generate cosmic-inspired shape with more details"""
    center_x = width // 2
    center_y = height // 2
    max_radius = min(width, height) * (0.3 + 0.2 * complexity)
    
    points = []
    num_turns = 3 + int(4 * complexity)
    num_points_per_turn = int(50 * complexity)
    
    for i in range(num_turns * num_points_per_turn):
        progress = i / (num_turns * num_points_per_turn)
        angle = 2 * math.pi * num_turns * progress
        radius = max_radius * progress
        radius_var = 0.7 + 0.6 * random.random()
        angle_var = angle + (random.random() - 0.5) * math.pi / 3
        x = center_x + radius * radius_var * math.cos(angle_var)
        y = center_y + radius * radius_var * math.sin(angle_var)
        points.append((x, y))
    
    num_clusters = int(8 + 12 * complexity)
    for _ in range(num_clusters):
        cluster_x = random.randint(int(width*0.1), int(width*0.9))
        cluster_y = random.randint(int(height*0.1), int(height*0.9))
        cluster_size = int(30 + 100 * complexity * random.random())
        num_stars = int(10 + 30 * complexity)
        for _ in range(num_stars):
            angle = random.random() * 2 * math.pi
            distance = cluster_size * random.random() * random.random()
            x = cluster_x + distance * math.cos(angle)
            y = cluster_y + distance * math.sin(angle)
            points.append((x, y))
    
    num_nebulae = int(3 + 5 * complexity)
    for _ in range(num_nebulae):
        nebula_x = random.randint(int(width*0.1), int(width*0.9))
        nebula_y = random.randint(int(height*0.1), int(height*0.9))
        nebula_size = int(50 + 150 * complexity)
        num_cloud_points = int(50 + 150 * complexity)
        for _ in range(num_cloud_points):
            angle = random.random() * 2 * math.pi
            distance = nebula_size * random.random() * 0.7
            x = nebula_x + distance * math.cos(angle)
            y = nebula_y + distance * math.sin(angle)
            points.append((x, y))
    
    return points

def generate_chaotic_shape(width, height, complexity=0.7):
    """Generate chaotic shape with more complexity"""
    points = []
    num_points = int(200 + 800 * complexity)
    
    for walk in range(int(2 + 3 * complexity)):
        x, y = random.randint(0, width), random.randint(0, height)
        for _ in range(num_points // int(2 + 3 * complexity)):
            points.append((x, y))
            angle = random.random() * 2 * math.pi
            distance = 1 + 15 * complexity * random.random()
            x += distance * math.cos(angle)
            y += distance * math.sin(angle)
            x = max(0, min(width, x))
            y = max(0, min(height, y))
    
    return points

def generate_fractal_shape(width, height, complexity=0.7):
    """Generate fractal-inspired recursive shape"""
    center_x = width // 2
    center_y = height // 2
    max_radius = min(width, height) * (0.2 + 0.3 * complexity)
    
    def recursive_branch(x, y, angle, depth, max_depth, branch_length):
        if depth > max_depth:
            return []
        
        points = []
        num_segments = int(3 + 7 * complexity)
        for i in range(num_segments):
            progress = i / num_segments
            px = x + branch_length * progress * math.cos(angle)
            py = y + branch_length * progress * math.sin(angle)
            points.append((px, py))
            
            if depth < max_depth and random.random() < 0.7:
                branch_angle = angle + (random.random() - 0.5) * math.pi / 2
                sub_length = branch_length * (0.4 + 0.3 * random.random())
                points.extend(recursive_branch(px, py, branch_angle, depth+1, max_depth, sub_length))
        
        return points
    
    max_depth = int(2 + 4 * complexity)
    initial_angle = random.random() * 2 * math.pi
    branch_length = max_radius * (0.5 + 0.5 * random.random())
    points = recursive_branch(center_x, center_y, initial_angle, 0, max_depth, branch_length)
    
    num_center_points = int(50 + 100 * complexity)
    for _ in range(num_center_points):
        angle = random.random() * 2 * math.pi
        distance = max_radius * 0.1 * random.random()
        x = center_x + distance * math.cos(angle)
        y = center_y + distance * math.sin(angle)
        points.append((x, y))
    
    return points

def generate_crystalline_shape(width, height, complexity=0.7):
    """Generate sharp crystalline shapes"""
    center_x = width // 2
    center_y = height // 2
    max_radius = min(width, height) * (0.3 + 0.3 * complexity)
    
    points = []
    num_crystals = int(5 + 15 * complexity)
    
    for _ in range(num_crystals):
        crystal_x = center_x + (random.random() - 0.5) * width * 0.8
        crystal_y = center_y + (random.random() - 0.5) * height * 0.8
        crystal_size = max_radius * (0.1 + 0.4 * random.random())
        num_points = random.randint(5, 10)
        crystal_points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points + random.random() * math.pi/6
            distance = crystal_size * (0.7 + 0.6 * random.random())
            x = crystal_x + distance * math.cos(angle)
            y = crystal_y + distance * math.sin(angle)
            crystal_points.append((x, y))
        points.extend(crystal_points)
        
        if complexity > 0.5:
            num_internal = int(3 + 7 * complexity)
            for _ in range(num_internal):
                internal_x = crystal_x + (random.random() - 0.5) * crystal_size * 0.5
                internal_y = crystal_y + (random.random() - 0.5) * crystal_size * 0.5
                internal_size = crystal_size * (0.1 + 0.3 * random.random())
                num_internal_points = random.randint(3, 6)
                for j in range(num_internal_points):
                    angle = 2 * math.pi * j / num_internal_points
                    distance = internal_size * (0.8 + 0.4 * random.random())
                    x = internal_x + distance * math.cos(angle)
                    y = internal_y + distance * math.sin(angle)
                    points.append((x, y))
    
    return points

def generate_complex_art(text, width=1024, height=1024):
    """Generate complex abstract art based on text input with AI analysis"""
    # Metin analizi - AI kullanarak
    ai_analysis = ai_text_analysis(text)
    
    # Duygu skorunu çıkar
    sentiment_score = 0
    for item in ai_analysis["sentiment"]:
        if item['label'] == 'POSITIVE':
            sentiment_score = item['score']
        elif item['label'] == 'NEGATIVE':
            sentiment_score = -item['score']
    
    # Karmaşıklık hesapla
    words = re.sub(r'[^a-zA-Z0-9ğüşıöçĞÜŞİÖÇ\s]', '', text).lower().split()
    complexity = min(1.0, max(0.3, len(set(words)) / 15))
    
    # Palet oluştur
    palette = generate_palette(text, ai_analysis)
    
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
        ratio = y / height
        r = int(bg_color1[0] * (1 - ratio) + bg_color2[0] * ratio)
        g = int(bg_color1[1] * (1 - ratio) + bg_color2[1] * ratio)
        b = int(bg_color1[2] * (1 - ratio) + bg_color2[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # AI'dan gelen temalara göre şekil jeneratörlerini seç
    shape_generators = []
    theme_labels = ai_analysis["themes"].get("labels", ["soyut"])
    
    if "doğa" in theme_labels or "organik" in theme_labels:
        shape_generators.append(generate_organic_shape)
    if "şehir" in theme_labels or "mekanik" in theme_labels:
        shape_generators.append(generate_geometric_shape)
    if "uzay" in theme_labels:
        shape_generators.append(generate_cosmic_shape)
    if "duygusal" in theme_labels or "kaotik" in theme_labels:
        shape_generators.append(generate_chaotic_shape)
    if "soyut" in theme_labels or "fraktal" in theme_labels:
        shape_generators.append(generate_fractal_shape)
    if "kristal" in theme_labels:
        shape_generators.append(generate_crystalline_shape)
    
    if not shape_generators:
        shape_generators = [generate_organic_shape, generate_geometric_shape, 
                            generate_cosmic_shape, generate_chaotic_shape,
                            generate_fractal_shape, generate_crystalline_shape]
    
    num_layers = int(5 + 8 * complexity)
    for layer in range(num_layers):
        layer_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer_img)
        shape_gen = random.choice(shape_generators)
        num_shapes = int(3 + 8 * complexity)
        
        for _ in range(num_shapes):
            center_x = random.randint(int(width*0.1), int(width*0.9))
            center_y = random.randint(int(height*0.1), int(height*0.9))
            shape_width = int(width * (0.1 + 0.6 * random.random()))
            shape_height = int(height * (0.1 + 0.6 * random.random()))
            points = shape_gen(shape_width, shape_height, complexity)
            
            min_x = min(p[0] for p in points) if points else 0
            min_y = min(p[1] for p in points) if points else 0
            max_x = max(p[0] for p in points) if points else shape_width
            max_y = max(p[1] for p in points) if points else shape_height
            
            scale_x = shape_width / (max_x - min_x) if max_x != min_x else 1
            scale_y = shape_height / (max_y - min_y) if max_y != min_y else 1
            
            scaled_points = []
            for x, y in points:
                scaled_x = center_x + (x - min_x) * scale_x
                scaled_y = center_y + (y - min_y) * scale_y
                scaled_points.append((scaled_x, scaled_y))
            
            color = random.choice(palette)
            alpha = int(30 + 200 * random.random() * complexity)
            fill_color = color + (alpha,)
            
            if len(scaled_points) > 2:
                layer_draw.polygon(scaled_points, fill=fill_color)
        
        if random.random() > 0.3:
            layer_img = layer_img.rotate(
                random.randint(-45, 45), 
                resample=Image.BICUBIC, 
                expand=False
            )
        
        if complexity > 0.6 and random.random() > 0.7:
            wave_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            wave_draw = ImageDraw.Draw(wave_img)
            for y_pos in range(0, height, 10):
                offset = int(5 * math.sin(y_pos / 20))
                wave_draw.line([(0, y_pos), (width, y_pos)], fill=(255, 255, 255, 10), width=2)
            layer_img = ImageChops.add(layer_img, wave_img)
        
        img = Image.alpha_composite(img.convert('RGBA'), layer_img)
    
    img = img.convert('RGB')
    
    texture = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    texture_draw = ImageDraw.Draw(texture)
    num_texture_points = int(8000 * complexity)
    
    for _ in range(num_texture_points):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(1, 8)
        color = random.choice(palette)
        alpha = random.randint(10, 100)
        texture_draw.ellipse([x, y, x+size, y+size], fill=color + (alpha,))
    
    img = Image.alpha_composite(img.convert('RGBA'), texture).convert('RGB')
    
    if sentiment_score > 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.4)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)
    elif sentiment_score < -0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.7)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    
    if "uzay" in theme_labels and random.random() > 0.4:
        glow = img.filter(ImageFilter.GaussianBlur(radius=10))
        enhancer = ImageEnhance.Brightness(glow)
        glow = enhancer.enhance(1.5)
        img = Image.blend(img, glow, 0.3)
    
    return img, ai_analysis

# 4) UI
st.title("İç ve Dış Dünyalarımızın Soyut Sanatı")
st.info("ℹ️ Metin girişlerinize göre otomatik olarak oluşturulan karmaşık soyut sanat eserleri (1024px çözünürlük)")

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
    .stTextArea textarea::placeholder {
        color: #888;
        font-style: italic;
    }
    .metric-explanation {
        font-size: 0.85rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 15px;
    }
    .ai-analysis-box {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="text-box">', unsafe_allow_html=True)
    inner_txt = st.text_area(
        "📖 İç Dünya (Duygu, düşünce veya rüyalarınızı birkaç cümle ile anlatın):", 
        height=120, 
        value="",
        placeholder="Örnek: Rüyalarımda gördüğüm renkli dünya, sonsuz olasılıklar, neşeli kaos ve organik formlar"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
with col2:
    st.markdown('<div class="text-box">', unsafe_allow_html=True)
    outer_txt = st.text_area(
        "🌍 Dış Dünya (Çevrenizde olan şeyleri, duyduğunuz, gördüğünüz, dokunduğunuz vs. şeyleri anlatın):", 
        height=120, 
        value="",
        placeholder="Örnek: Şehirdeki gri binalar, trafik karmaşası, sistematik düzen ve geometrik yapılar"
    )
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🎨 Oluştur ve Karşılaştır", use_container_width=True):
    if not inner_txt or not outer_txt:
        st.warning("⚠️ Lütfen her iki metni de girin.")
        st.stop()

    with st.spinner("🖼️ Yüksek çözünürlüklü soyut sanat eserleri oluşturuluyor..."):
        start_time = time.time()
        img1, ai_analysis1 = generate_complex_art(inner_txt)
        img2, ai_analysis2 = generate_complex_art(outer_txt)
        generation_time = time.time() - start_time

    st.success(f"✅ Sanat eserleri başarıyla oluşturuldu! Süre: {generation_time:.1f} saniye")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="art-title">İç Dünya</div>', unsafe_allow_html=True)
        st.image(img1, use_container_width=True)
        themes = ai_analysis1["themes"].get("labels", ["soyut"])
        st.markdown(f'<div class="art-analysis"><b>AI Analizi:</b> {", ".join(themes[:3])} temaları</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="art-title">Dış Dünya</div>', unsafe_allow_html=True)
        st.image(img2, use_container_width=True)
        themes = ai_analysis2["themes"].get("labels", ["soyut"])
        st.markdown(f'<div class="art-analysis"><b>AI Analizi:</b> {", ".join(themes[:3])} temaları</div>', unsafe_allow_html=True)

    st.subheader("🤖 Derinlemesine AI Analizi")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="ai-analysis-box">', unsafe_allow_html=True)
        st.markdown("**İç Dünya Analizi**")
        
        sentiment = ai_analysis1["sentiment"][0]
        st.write(f"**Duygu Durumu:** {sentiment['label']} (%{sentiment['score']*100:.1f})")
        
        st.write("**Temalar:**")
        for label, score in zip(ai_analysis1["themes"]["labels"], ai_analysis1["themes"]["scores"]):
            st.write(f"- {label} (%{score*100:.1f})")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="ai-analysis-box">', unsafe_allow_html=True)
        st.markdown("**Dış Dünya Analizi**")
        
        sentiment = ai_analysis2["sentiment"][0]
        st.write(f"**Duygu Durumu:** {sentiment['label']} (%{sentiment['score']*100:.1f})")
        
        st.write("**Temalar:**")
        for label, score in zip(ai_analysis2["themes"]["labels"], ai_analysis2["themes"]["scores"]):
            st.write(f"- {label} (%{score*100:.1f})")
        st.markdown('</div>', unsafe_allow_html=True)

    try:
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        m1 = compute_metrics(arr1)
        m2 = compute_metrics(arr2)
        iou = calculate_iou(m1, m2)
        
        similarity_color = "#4CAF50" if iou > 0.5 else "#FF9800" if iou > 0.3 else "#F44336"
        st.markdown(f"""
        <div style="background-color: #f0f0f0; border-radius: 5px; padding: 15px; margin: 20px 0; text-align: center;">
            <h3 style="color: #333; margin-bottom: 10px;">🔍 Görsel Benzerlik Analizi</h3>
            <div style="font-size: 2em; font-weight: bold; color: {similarity_color};">{iou:.3f}</div>
            <div style="margin-top: 10px; color: #666;">0 (tamamen farklı) - 1 (tamamen benzer)</div>
        </div>
        """, unsafe_allow_html=True)

        labels = ["Parlaklık", "Kontrast", "Renk Canlılığı", "Karmaşıklık"]
        
        st.markdown("""
        <div class="metric-explanation">
            <b>Metrik Açıklamaları:</b><br>
            <b>Parlaklık:</b> Görselin ortalama aydınlık seviyesi<br>
            <b>Kontrast:</b> Renk ve tonlar arasındaki farklılıklar<br>
            <b>Renk Canlılığı:</b> Renklerin doygunluk ve çeşitliliği<br>
            <b>Karmaşıklık:</b> Görseldeki detay ve desen zenginliği
        </div>
        """, unsafe_allow_html=True)
        
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
        ax.set_rlabel_position(180)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        st.pyplot(fig)
        plt.close(fig)
        
        metric_data = {
            "Metrik": labels,
            "İç Dünya": [f"{x:.2f}" for x in m1],
            "Dış Dünya": [f"{x:.2f}" for x in m2],
            "Fark": [f"{abs(a-b):.2f}" for a, b in zip(m1, m2)]
        }
        st.subheader("📊 Metrik Değerleri")
        st.table(metric_data)
        
    except Exception as e:
        st.error(f"❌ Analiz hatası: {str(e)}")