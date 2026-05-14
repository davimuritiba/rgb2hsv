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
