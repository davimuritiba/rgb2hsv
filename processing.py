from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def carregar_rgb_do_disco(caminho: str) -> np.ndarray:
    img = Image.open(caminho).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def rgb_para_hsv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def hsv_para_preview_matiz(hsv: np.ndarray) -> np.ndarray:
    """RGB uint8: matiz (H) com saturação e valor máximos — leitura visual do canal H."""
    h, _, _ = cv2.split(hsv)
    s = np.full(h.shape, 255, dtype=np.uint8)
    v = np.full(h.shape, 255, dtype=np.uint8)
    hsv_max = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_max, cv2.COLOR_HSV2RGB)


def hsv_para_preview_saturacao(hsv: np.ndarray) -> np.ndarray:
    """RGB uint8: saturação (S) com matiz e valor máximos."""
    _, s, _ = cv2.split(hsv)
    h = np.full(s.shape, 0, dtype=np.uint8)  # Matiz neutro (cinza)
    v = np.full(s.shape, 255, dtype=np.uint8)
    hsv_max = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_max, cv2.COLOR_HSV2RGB)


def hsv_para_preview_valor(hsv: np.ndarray) -> np.ndarray:
    """RGB uint8: valor (V) em escala de cinza."""
    _, _, v = cv2.split(hsv)
    # Converter para grayscale: replicar o canal V em RGB
    return cv2.cvtColor(cv2.merge([v, v, v]), cv2.COLOR_BGR2RGB)


def hsv_para_canal_cinza(hsv: np.ndarray, canal: int) -> np.ndarray:
    """
    Extrai um canal HSV e retorna como imagem cinza (RGB com 3 canais iguais).
    canal: 0=H, 1=S, 2=V
    """
    canais = cv2.split(hsv)
    c = canais[canal]
    # Normalizar H (0-180) para 0-255 para visualização melhor
    if canal == 0:
        c = cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(cv2.merge([c, c, c]), cv2.COLOR_RGB2RGB)


def ajustar_hsv(hsv: np.ndarray, h_offset: int = 0, s_factor: float = 1.0, v_factor: float = 1.0) -> np.ndarray:
    """
    Ajusta os canais H, S, V da imagem HSV.
    
    h_offset: deslocamento de matiz (-180 a 180)
    s_factor: multiplicador de saturação (0.0 a 2.0)
    v_factor: multiplicador de valor/brilho (0.0 a 2.0)
    """
    h, s, v = cv2.split(hsv)
    
    # Ajustar H com wrap-around
    h = ((h.astype(np.int32) + h_offset) % 180).astype(np.uint8)
    
    # Ajustar S
    s = np.clip(s.astype(np.float32) * s_factor, 0, 255).astype(np.uint8)
    
    # Ajustar V
    v = np.clip(v.astype(np.float32) * v_factor, 0, 255).astype(np.uint8)
    
    return cv2.merge([h, s, v])


def aplicar_blur_gaussiano(hsv: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Aplica blur Gaussiano. kernel_size deve ser ímpar (3, 5, 7, ...)"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(hsv, (kernel_size, kernel_size), 0)


def aplicar_blur_mediano(hsv: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Aplica blur mediano (melhor para ruído)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(hsv, kernel_size)


def detectar_bordas_canny(rgb: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
    """Detecta bordas usando Canny. Retorna imagem em escala de cinza."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def detectar_bordas_sobel(rgb: np.ndarray) -> np.ndarray:
    """Detecta bordas usando Sobel. Retorna imagem em escala de cinza."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def gerar_histograma(hsv: np.ndarray, canal: int = 0, bins: int = 180) -> np.ndarray:
    """
    Gera histograma de um canal HSV.
    canal: 0=H (matiz), 1=S (saturação), 2=V (valor)
    Retorna imagem do histograma (RGB).
    """
    canais = cv2.split(hsv)
    hist = cv2.calcHist([canais[canal]], [0], None, [bins], 
                        [0, 256] if canal > 0 else [0, 180])
    
    # Normalizar histograma para 0-200 pixels
    hist = cv2.normalize(hist, hist, 0, 200, cv2.NORM_MINMAX).flatten()
    
    # Criar imagem do histograma com tamanho maior
    hist_height = 250
    hist_width = max(400, bins)  # Mínimo 400 pixels de largura
    hist_img = np.ones((hist_height, hist_width, 3), dtype=np.uint8) * 255
    
    # Calcular largura de cada barra
    bar_width = max(1, hist_width // bins)
    
    # Desenhar barras
    for i in range(bins):
        h = int(hist[i])
        x_start = i * bar_width
        x_end = min(x_start + bar_width, hist_width)
        cv2.rectangle(hist_img, (x_start, hist_height - h), (x_end - 1, hist_height), 
                     (100, 100, 100), -1)
    
    # Converter de BGR para RGB
    return cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB)
