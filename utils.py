# utils.py
import io
import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def image_bytes_to_rgb_numpy(image_bytes):
    """
    Mengkonversi data byte gambar (misalnya dari unggahan Streamlit) 
    menjadi array NumPy dengan format warna RGB.
    """
    if not image_bytes:
        logger.warning("image_bytes_to_rgb_numpy dipanggil dengan image_bytes kosong.")
        return None
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil_image)
    except Exception as e:
        logger.error(f"Error saat mengkonversi image bytes ke NumPy array: {e}", exc_info=True)
        return None

def apply_clahe_enhancement(image_rgb: np.ndarray):
    """
    Menerapkan Contrast Limited Adaptive Histogram Equalization (CLAHE)
    pada gambar RGB untuk meningkatkan kontras lokal.
    """
    if image_rgb is None:
        logger.warning("apply_clahe_enhancement menerima input gambar None.")
        return None
    try:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            logger.error("Input untuk CLAHE harus berupa gambar RGB (3 channel).")
            return image_rgb 

        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(image_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel) 
        enhanced_lab_image = cv2.merge((cl, a_channel, b_channel))
        enhanced_rgb_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2RGB)
        logger.info("Enhancement CLAHE berhasil diterapkan.")
        return enhanced_rgb_image
    except cv2.error as e_cv: 
        logger.error(f"Error OpenCV saat menerapkan CLAHE: {e_cv}", exc_info=True)
        return image_rgb 
    except Exception as e: 
        logger.error(f"Error umum saat menerapkan CLAHE: {e}", exc_info=True)
        return image_rgb 