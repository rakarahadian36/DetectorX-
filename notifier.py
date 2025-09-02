# notifier.py
import requests
import os
from dotenv import load_dotenv
from imgurpython import ImgurClient
from imgurpython.helpers.error import ImgurClientError, ImgurClientRateLimitError 
import logging
import urllib.parse 

logger = logging.getLogger(__name__)
load_dotenv() 

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CALLMEBOT_API_KEY = os.getenv('CALLMEBOT_API_KEY')
RECEIVER_WHATSAPP_NUMBER = os.getenv("RECEIVER_WHATSAPP_NUMBER") 
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")

imgur_client_notifier = None
if IMGUR_CLIENT_ID:
    try:
        imgur_client_notifier = ImgurClient(IMGUR_CLIENT_ID, None) 
        logger.info("Klien Imgur untuk notifier.py berhasil diinisialisasi.")
    except Exception as e:
        logger.error(f"Gagal menginisialisasi klien Imgur: {e}")
else:
    logger.warning("IMGUR_CLIENT_ID tidak ditemukan di .env. Fungsi unggah ke Imgur tidak akan aktif.")

def upload_to_imgur(image_path):
    """Mengunggah gambar ke Imgur dan mengembalikan URL publiknya."""
    if not imgur_client_notifier:
        logger.warning("Klien Imgur tidak aktif atau tidak terkonfigurasi, unggahan dilewati.")
        return None
    if not os.path.exists(image_path):
        logger.error(f"File gambar untuk diunggah ke Imgur tidak ditemukan: {image_path}")
        return None
    try:
        logger.info(f"Mengunggah file '{image_path}' ke Imgur...")
        image_obj = imgur_client_notifier.upload_from_path(image_path, anon=True)
        imgur_link = image_obj['link']
        logger.info(f"Berhasil diunggah ke Imgur. Link: {imgur_link}")
        return imgur_link
    except ImgurClientRateLimitError as re: 
        logger.error(f"Imgur API rate limit terlampaui: {re}", exc_info=True)
        return None
    except ImgurClientError as ice: 
        logger.error(f"Error dari Imgur API (ImgurClientError) saat unggah '{image_path}': {ice}", exc_info=True)
        if hasattr(ice, 'status_code') and hasattr(ice, 'error_message'):
            logger.error(f"Detail Error Imgur: Status {ice.status_code}, Pesan: {ice.error_message}")
        return None
    except Exception as e: 
        logger.error(f"Error tidak terduga saat mengunggah '{image_path}' ke Imgur: {e}", exc_info=True)
        return None

def format_rich_notification_message(message_details: dict, imgur_link_for_message: str = None):
    """
    Mempersiapkan teks pesan notifikasi agar sesuai dengan format yang diinginkan pengguna.
    Contoh: "LIVE ALERT! Terdeteksi Fire di Hutan dan lahan dari Unggahan Video: namafile.mp4"
    """
    det_type = message_details.get('detection_type', 'Bahaya').capitalize()
    location = message_details.get('location', 'Lokasi Tidak Diketahui')
    source_info = message_details.get('source_info', 'sumber tidak diketahui')
    confidence = message_details.get('confidence', 0.0)
    timestamp = message_details.get('timestamp', 'Waktu Tidak Tercatat')
    gemini_text = message_details.get('gemini_analysis')

    # 1. Membuat Judul Notifikasi
    # Menggabungkan semua informasi ke dalam satu baris judul
    title_line = f"🔥🚨 LIVE ALERT! Terdeteksi {det_type} di {location} dari {source_info}"

    # 2. Membuat Detail Deteksi
    # Menggunakan format desimal untuk kepercayaan dan menghapus emoji
    confidence_text = f"Kepercayaan: {confidence:.2f}"
    timestamp_text = f"Waktu Deteksi: {timestamp}"
    detection_details_text = f"{confidence_text}\n{timestamp_text}"

    # 3. Merakit pesan utama
    full_message = f"{title_line}\n\n{detection_details_text}"

    # 4. Menambahkan link gambar dalam satu baris
    if imgur_link_for_message:
        full_message += f"\n\n🖼️ Frame Deteksi: {imgur_link_for_message}"

    # 5. Menambahkan Analisis Gemini dengan header yang sesuai
    if gemini_text and \
       gemini_text.strip().lower() not in ["analisis gemini tidak tersedia.",
                                           "file gambar tidak ditemukan untuk analisis.",
                                           "analisis gemini tidak menghasilkan output yang diharapkan.",
                                           "analisis gemini diblokir."]:
        gemini_header = "\n\n— 🧠 Analisis & Saran Gemini AI —"
        # Langsung gabungkan teks analisis dari Gemini
        full_message += f"{gemini_header}\n{gemini_text.strip()}"

    return full_message.strip()


def send_telegram_notification(message_details: dict, image_path_annotated: str = None):
    """
    Mengirim notifikasi ke Telegram dengan gambar hasil deteksi (anotasi) dan detail lengkap.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Token Bot Telegram atau Chat ID tidak lengkap. Notifikasi Telegram dibatalkan.")
        return False
    
    imgur_link_tg = None
    if image_path_annotated and os.path.exists(image_path_annotated):
        imgur_link_tg = upload_to_imgur(image_path_annotated) 
        if not imgur_link_tg:
            logger.warning("Telegram: Gagal unggah ke Imgur, link tidak akan disertakan di caption jika pengiriman file langsung juga gagal.")

    full_caption_text = format_rich_notification_message(message_details, imgur_link_for_message=imgur_link_tg)

    try:
        if image_path_annotated and os.path.exists(image_path_annotated):
            logger.info(f"Telegram: Mencoba mengirim file gambar anotasi '{image_path_annotated}' langsung...")
            url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path_annotated, 'rb') as photo_file:
                files = {'photo': photo_file}
                data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': full_caption_text, 'parse_mode': 'Markdown'}
                response_photo = requests.post(url_photo, files=files, data=data, timeout=30) 
                response_photo.raise_for_status() 
            logger.info(f"Notifikasi gambar anotasi '{image_path_annotated}' dengan detail berhasil terkirim ke Telegram.")
            return True
        else: 
            final_text_to_send = full_caption_text
            if image_path_annotated:
                logger.warning(f"Telegram: File gambar anotasi tidak ditemukan di '{image_path_annotated}'.")
                final_text_to_send += "\n\n(Peringatan: Gambar deteksi tidak dapat disertakan.)" 
            else:
                logger.info("Telegram: Tidak ada path gambar anotasi. Mengirim notifikasi teks saja.")
            
            url_msg_txt = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload_msg_txt = {'chat_id': TELEGRAM_CHAT_ID, 'text': final_text_to_send, 'parse_mode': 'Markdown'}
            response_txt = requests.post(url_msg_txt, data=payload_msg_txt, timeout=10)
            response_txt.raise_for_status()
            logger.info("Notifikasi teks Telegram berhasil terkirim.")
            return True
            
    except requests.exceptions.RequestException as e: 
        logger.error(f"Error saat mengirim notifikasi Telegram (upaya utama): {e}")
        if hasattr(e, 'response') and e.response is not None: 
            logger.error(f"Respons API Telegram: Status {e.response.status_code}, Isi: {e.response.text}")
        try:
            logger.warning("Telegram: Upaya utama pengiriman notifikasi gagal, mencoba mengirim pesan teks lengkap sebagai fallback...")
            text_for_fallback = full_caption_text
            if image_path_annotated and os.path.exists(image_path_annotated):
                 text_for_fallback += "\n\n(Peringatan: Gagal mengirim gambar deteksi secara langsung ke Telegram.)"

            url_msg_txt_fallback = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload_msg_txt_fallback = {'chat_id': TELEGRAM_CHAT_ID, 'text': text_for_fallback, 'parse_mode': 'Markdown'}
            requests.post(url_msg_txt_fallback, data=payload_msg_txt_fallback, timeout=10).raise_for_status()
            logger.info("Pesan teks fallback Telegram berhasil terkirim setelah kegagalan.")
            return True
        except Exception as ef:
             logger.error(f"Gagal mengirim pesan teks fallback Telegram: {ef}")
        return False 
    except Exception as e_gen: 
        logger.error(f"Error tidak terduga di send_telegram_notification: {e_gen}", exc_info=True)
        return False

def send_whatsapp_notification(message_details: dict, image_path_annotated: str = None):
    """
    Mengirim notifikasi WhatsApp menggunakan CallMeBot dengan gambar hasil deteksi (via Imgur) dan detail lengkap.
    """
    if not CALLMEBOT_API_KEY or not RECEIVER_WHATSAPP_NUMBER:
        logger.warning("API Key CallMeBot atau Nomor WhatsApp Penerima tidak lengkap. Notifikasi WhatsApp dilewati.")
        return False

    logger.info(f"Mencoba mengirim notifikasi WhatsApp via CallMeBot ke nomor: {RECEIVER_WHATSAPP_NUMBER}...")
    
    phone_number_cleaned = RECEIVER_WHATSAPP_NUMBER.lstrip('+')
    
    imgur_link_for_wa = None
    if image_path_annotated and os.path.exists(image_path_annotated):
        logger.info(f"WhatsApp: Mencoba unggah '{image_path_annotated}' ke Imgur...")
        imgur_link_for_wa = upload_to_imgur(image_path_annotated)
    
    full_text_message_wa = format_rich_notification_message(message_details, imgur_link_for_message=imgur_link_for_wa)
    
    if imgur_link_for_wa is None and image_path_annotated and os.path.exists(image_path_annotated):
        full_text_message_wa += "\n\n(Info: Gagal mengunggah gambar deteksi ke Imgur untuk pratinjau WhatsApp.)"
    elif image_path_annotated and not os.path.exists(image_path_annotated): 
        logger.warning(f"WhatsApp: File gambar anotasi tidak ditemukan di '{image_path_annotated}'.")
        full_text_message_wa += "\n\n(Info: File gambar deteksi tidak tersedia untuk notifikasi ini.)"
    elif not image_path_annotated:
        logger.info("WhatsApp: Tidak ada path gambar anotasi. Notifikasi akan dikirim tanpa gambar.")

    try:
        encoded_full_message = urllib.parse.quote_plus(full_text_message_wa)
        callmebot_url = (f"https://api.callmebot.com/whatsapp.php?"
                          f"phone={phone_number_cleaned}"
                          f"&text={encoded_full_message}"
                          f"&apikey={CALLMEBOT_API_KEY}")
            
        logger.debug(f"CallMeBot URL: {callmebot_url}")
        response = requests.get(callmebot_url, timeout=20)
        response.raise_for_status() 
        logger.info(f"Notifikasi WhatsApp berhasil dikirim via CallMeBot. Respons server: {response.text[:100]}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Gagal mengirim pesan WhatsApp via CallMeBot: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Respons CallMeBot: Status {e.response.status_code}, Isi: {e.response.text}")
        return False
    except Exception as e_gen:
        logger.error(f"Error lain saat kirim WhatsApp via CallMeBot: {e_gen}", exc_info=True)
        return False