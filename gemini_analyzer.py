import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Inisialisasi logger untuk file ini
logger = logging.getLogger(__name__)

# Muat variabel lingkungan dari file .env
load_dotenv()

# --- BLOK KONFIGURASI GEMINI ---

# Gunakan konstanta untuk nama model agar mudah diganti di masa mendatang.
# 'gemini-1.5-flash-latest' adalah pilihan yang efisien untuk analisis gambar.
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'

# Ambil API Key dari environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Variabel untuk menampung instance model yang sudah diinisialisasi
gemini_model_vision = None

# Lakukan konfigurasi hanya jika API Key tersedia
if GEMINI_API_KEY:
    try:
        # Konfigurasi API Google
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Inisialisasi model Generatif
        gemini_model_vision = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        logger.info(f"Model Gemini '{GEMINI_MODEL_NAME}' berhasil diinisialisasi.")
        
    except Exception as e:
        # Tangani error jika inisialisasi gagal
        logger.error(f"Gagal mengkonfigurasi atau memuat model Gemini: {e}", exc_info=True)
        # Set API Key ke None agar fungsi analisis tidak akan dijalankan
        GEMINI_API_KEY = None
else:
    # Beri peringatan jika API Key tidak ditemukan
    logger.warning("GEMINI_API_KEY tidak ditemukan di file .env. Fungsi analisis Gemini AI tidak akan aktif.")

# --- FUNGSI UTAMA ---

def analyze_image_with_gemini(image_path: str, prompt_text: str) -> str:
    """
    Menganalisis gambar menggunakan model multimodal Gemini (Vision).

    Args:
        image_path (str): Path lengkap ke file gambar yang akan dianalisis.
        prompt_text (str): Teks prompt yang akan digunakan untuk memandu analisis.

    Returns:
        str: Hasil analisis dari Gemini AI, atau pesan error/peringatan.
    """
    # Validasi awal: Pastikan API Key dan model sudah siap
    if not GEMINI_API_KEY or not gemini_model_vision:
        logger.warning("Analisis Gemini tidak dapat dilanjutkan karena API Key atau Model tidak dikonfigurasi.")
        return "Analisis Gemini tidak tersedia."

    # Validasi keberadaan file gambar
    if not os.path.exists(image_path):
        logger.error(f"File gambar untuk analisis Gemini tidak ditemukan di path: {image_path}")
        return f"Analisis Gagal: File gambar tidak ditemukan di {image_path}"

    img_pil = None
    try:
        logger.info(f"Membuka gambar '{image_path}' untuk dianalisis oleh Gemini...")
        
        # Buka gambar menggunakan Pillow
        img_pil = Image.open(image_path)
        
        # Kirim prompt (teks dan gambar) ke model Gemini
        response = gemini_model_vision.generate_content([prompt_text, img_pil])
        
        # Ekstraksi hasil teks dari respons dengan aman
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            analysis_text = ''.join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        else:
            analysis_text = ""

        if analysis_text:
            logger.info("Analisis dari Gemini AI berhasil diterima.")
            return analysis_text.strip()
            
        # Periksa apakah prompt diblokir karena alasan keamanan
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason
            msg = getattr(response.prompt_feedback, 'block_reason_message', "Tidak ada pesan detail.")
            logger.warning(f"Analisis Gemini AI diblokir. Alasan: {reason}. Pesan: {msg}")
            return f"Analisis Gemini AI diblokir: {reason}. {msg}"
            
        else:
            # Jika tidak ada teks dan tidak diblokir, ini adalah kasus yang tidak terduga
            logger.warning(f"Analisis Gemini AI tidak menghasilkan output teks. Respons (sebagian): {str(response)[:250]}")
            return "Analisis Gemini AI tidak menghasilkan output yang diharapkan."
            
    except Exception as e:
        logger.error(f"Terjadi error saat melakukan analisis dengan Gemini AI: {str(e)}", exc_info=True)
        return f"Error saat berkomunikasi dengan Gemini AI: {str(e)}"
        
    finally:
        # Pastikan objek gambar Pillow ditutup untuk melepaskan memori
        if img_pil and hasattr(img_pil, 'close'): 
            try:
                img_pil.close()
                logger.debug(f"Objek PIL Image untuk '{image_path}' telah ditutup.")
            except Exception as e_close:
                logger.error(f"Gagal menutup objek PIL Image: {e_close}")