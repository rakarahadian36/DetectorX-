# run.py
import cv2
import argparse
import os
import time
import logging
import tempfile
from PIL import Image as PILImage
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import gc 
from datetime import datetime 

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from detector import YoloDetector 
    logger.debug("Modul 'detector' dan kelas 'YoloDetector' berhasil diimpor.")
except ImportError as e:
    logger.error(f"Gagal mengimpor YoloDetector dari detector.py: {e}. Pastikan file dan nama kelas sudah benar.")
    exit(1) 

try:
    from notifier import send_telegram_notification, send_whatsapp_notification 
    logger.debug("Fungsi-fungsi dari modul 'notifier' berhasil diimpor.")
except ImportError as e:
    logger.error(f"Gagal mengimpor dari notifier.py: {e}.")
    exit(1)

try:
    from utils import apply_clahe_enhancement
    logger.debug("Fungsi dari modul 'utils' berhasil diimpor.")
except ImportError as e:
    logger.error(f"Gagal mengimpor dari utils.py: {e}.")
    exit(1)

try:
    from gemini_analyzer import analyze_image_with_gemini, GEMINI_API_KEY as GEMINI_AVAILABLE_CLI 
    logger.debug("Fungsi dari modul 'gemini_analyzer' berhasil diimpor.")
except ImportError as e:
    logger.error(f"Gagal mengimpor dari gemini_analyzer.py: {e}.")
    exit(1)


load_dotenv() 
logger.info("Variabel lingkungan dari .env (jika ada) telah dimuat.")

last_notification_times_cli = {}

def attempt_remove_temp_file_cli(file_path, max_retries=3, delay=0.1):
    if not file_path or not os.path.exists(file_path):
        logger.debug(f"CLI: Penghapusan file temporer dilewati, path tidak valid atau file tidak ada ({file_path})")
        return
    retries = 0
    while retries < max_retries:
        try:
            os.remove(file_path)
            logger.info(f"CLI: File temporer '{file_path}' berhasil dihapus pada percobaan ke-{retries+1}.")
            return 
        except PermissionError as pe:
            retries += 1
            logger.warning(f"CLI: Percobaan ke-{retries} gagal menghapus '{file_path}': {pe}. Menunggu {delay} detik.")
            if retries < max_retries: time.sleep(delay)
            else: logger.error(f"CLI: Gagal menghapus file temporer '{file_path}' setelah {max_retries} percobaan karena PermissionError: {pe}.")
        except Exception as e:
            logger.error(f"CLI: Error tidak terduga saat mencoba menghapus file temporer '{file_path}': {e}", exc_info=True)
            return

def process_video_source_cli(
    source_str, model_path_cli, 
    confidence_cli, iou_cli, imgsz_cli, augment_cli,
    use_clahe_cli, notification_cooldown_cli,
    enable_telegram_cli, enable_whatsapp_cli, 
    enable_gemini_cli, location_name_cli
    ):
    logger.info(f"==> Memulai fungsi 'process_video_source_cli' untuk sumber: {source_str}")
    global last_notification_times_cli 
    
    detector_cli = YoloDetector(model_path=model_path_cli)
    if not detector_cli.model: 
        logger.error(f"Gagal memuat model YOLO dari '{model_path_cli}' di dalam process_video_source_cli. Proses CLI dihentikan.")
        return

    logger.info(f"Memulai pemrosesan sumber: '{source_str}' dengan model '{model_path_cli}'")
    logger.info(f"Parameter Deteksi DIGUNAKAN: Confidence={confidence_cli}, IoU={iou_cli}, ImgSz={imgsz_cli}, Augment={augment_cli}, CLAHE={use_clahe_cli}")
    logger.info(f"Parameter Notifikasi: Cooldown={notification_cooldown_cli}s, Telegram={enable_telegram_cli}, WhatsApp={enable_whatsapp_cli}")
    logger.info(f"Analisis Gemini AI: {enable_gemini_cli}, Lokasi: {location_name_cli}")
    
    is_camera_source = source_str.lower() == 'camera' or source_str.isdigit()
    capture_source_value = int(source_str) if source_str.isdigit() else (0 if source_str.lower() == 'camera' else source_str)
    
    source_info_for_notif = ""
    if is_camera_source:
        source_info_for_notif = f"dari Kamera ID: {capture_source_value}"
    else:
        source_info_for_notif = f"dari File: {os.path.basename(source_str)}"


    logger.info(f"Mencoba membuka sumber capture: {capture_source_value}")
    video_capture_cli = cv2.VideoCapture(capture_source_value)
    if not video_capture_cli.isOpened():
        logger.error(f"Tidak dapat membuka sumber video/kamera: '{source_str}' (diproses sebagai: {capture_source_value}). Pastikan sumber tersedia dan path benar.")
        return

    logger.info(f"Sumber video/kamera '{capture_source_value}' berhasil dibuka.")
    cv2.namedWindow("Deteksi Api & Asap - Mode CLI", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    try:
        while video_capture_cli.isOpened():
            frame_count += 1
            logger.debug(f"CLI: Membaca frame baru #{frame_count}...")
            ret, frame_bgr_stream = video_capture_cli.read() 
            if not ret:
                logger.info("Selesai memproses sumber video atau stream berakhir (tidak ada frame lagi).")
                break 
            
            logger.debug(f"CLI: Frame #{frame_count} berhasil dibaca. Memulai pra-pemrosesan dan deteksi...")
            frame_to_process_cli = frame_bgr_stream.copy() 
            if use_clahe_cli: 
                logger.debug(f"CLI: Menerapkan CLAHE pada frame #{frame_count}...")
                enhanced_rgb_frame = apply_clahe_enhancement(cv2.cvtColor(frame_to_process_cli, cv2.COLOR_BGR2RGB))
                if enhanced_rgb_frame is not None:
                    frame_to_process_cli = cv2.cvtColor(enhanced_rgb_frame, cv2.COLOR_RGB2BGR)
            
            detected_objects, annotated_frame_cli_output = detector_cli.detect(
                frame_to_process_cli, 
                confidence_threshold=confidence_cli, 
                iou_threshold=iou_cli,
                imgsz=imgsz_cli,
                augment=augment_cli
            )
            logger.info(f"CLI: Deteksi pada frame #{frame_count} selesai. Jumlah objek 'fire'/'smoke' yang lolos filter label: {len(detected_objects)}")
            cv2.imshow("Deteksi Api & Asap - Mode CLI", annotated_frame_cli_output)

            for det_obj in detected_objects: 
                label_detected = det_obj['label'].lower()
                confidence_score = det_obj['confidence']
                
                current_event_time = time.time()
                if (current_event_time - last_notification_times_cli.get(label_detected, 0)) > notification_cooldown_cli:
                    logger.info(f"===> NOTIFIKASI UNTUK LOKASI '{location_name_cli}': Jenis '{label_detected.upper()}' (Kepercayaan: {confidence_score:.2f}).")
                    
                    temp_annotated_image_path_cli = None 
                    fp_cli_annotated_obj = None 
                    pil_annotated_cli_for_saving = None 
                    gemini_analysis_result_cli = None
                    timestamp_cli_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    try:
                        pil_annotated_cli_for_saving = PILImage.fromarray(cv2.cvtColor(annotated_frame_cli_output, cv2.COLOR_BGR2RGB))
                        fp_cli_annotated_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", prefix=f"cli_annotated_{label_detected}_")
                        pil_annotated_cli_for_saving.save(fp_cli_annotated_obj, "JPEG")
                        temp_annotated_image_path_cli = fp_cli_annotated_obj.name
                        fp_cli_annotated_obj.close(); fp_cli_annotated_obj = None 
                        logger.debug(f"CLI: Frame ANOTASI disimpan ke file temporer: {temp_annotated_image_path_cli}")
                        
                        if enable_gemini_cli and GEMINI_AVAILABLE_CLI and temp_annotated_image_path_cli:
                            logger.info("CLI: Menganalisis gambar deteksi dengan Gemini AI...")
                            gemini_prompt_cli = (f"Gambar ini menunjukkan deteksi '{label_detected}' di lokasi '{location_name_cli}' {source_info_for_notif} dengan anotasi visual. "
                                                 f"Berikan analisis singkat mengenai potensi bahaya, kemungkinan penyebab, dan langkah keamanan dasar (poin-poin, emoji jika relevan). Ringkas.")
                            gemini_analysis_result_cli = analyze_image_with_gemini(temp_annotated_image_path_cli, gemini_prompt_cli)
                            logger.info(f"--- Hasil Analisis Gemini AI (CLI) ---\n{gemini_analysis_result_cli}\n------------------------------------")

                        # --- PERSIAPAN MESSAGE_DETAILS BARU UNTUK CLI ---
                        message_details_cli = {
                            'detection_type': label_detected.capitalize(),
                            'location': location_name_cli,
                            'source_info': source_info_for_notif,
                            'confidence': confidence_score,
                            'timestamp': timestamp_cli_str,
                            'gemini_analysis': gemini_analysis_result_cli
                        }
                        
                        if enable_telegram_cli:
                            logger.info("CLI: Mengirim notifikasi via Telegram...")
                            send_telegram_notification(message_details_cli, image_path_annotated=temp_annotated_image_path_cli)
                        
                        if enable_whatsapp_cli:
                            logger.info("CLI: Mengirim notifikasi via WhatsApp (CallMeBot)...")
                            send_whatsapp_notification(message_details_cli, image_path_annotated=temp_annotated_image_path_cli)
                        
                        last_notification_times_cli[label_detected] = current_event_time 

                    except Exception as e_notif_process_cli: 
                        logger.error(f"CLI: Terjadi error saat proses notifikasi/analisis Gemini: {e_notif_process_cli}", exc_info=True)
                    finally:
                        if hasattr(pil_annotated_cli_for_saving, 'close'): 
                            try: pil_annotated_cli_for_saving.close()
                            except: pass
                        if fp_cli_annotated_obj is not None : 
                            try: fp_cli_annotated_obj.close()
                            except: pass
                        gc.collect() 
                        if temp_annotated_image_path_cli: 
                            attempt_remove_temp_file_cli(temp_annotated_image_path_cli)
                else: 
                    logger.debug(f"CLI: Deteksi '{label_detected}' masih dalam masa cooldown notifikasi.")
            
            key_press_event = cv2.waitKey(1) & 0xFF 
            if key_press_event == ord('q') or key_press_event == 27: 
                logger.info("Perintah keluar (q/ESC) diterima dari keyboard. Menghentikan proses CLI.")
                break
            try: 
                if cv2.getWindowProperty("Deteksi Api & Asap - Mode CLI", cv2.WND_PROP_VISIBLE) < 1: 
                    logger.info("Jendela tampilan ('Deteksi Api & Asap - Mode CLI') ditutup oleh pengguna. Menghentikan proses CLI.")
                    break
            except cv2.error: 
                logger.info("Jendela tampilan sudah tidak ada (kemungkinan ditutup paksa). Menghentikan proses CLI.")
                break
    finally:
        if video_capture_cli and video_capture_cli.isOpened():
            video_capture_cli.release() 
        cv2.destroyAllWindows() 
        logger.info("Sumber video CLI telah dilepaskan dan semua jendela OpenCV ditutup.")

if __name__ == "__main__":
    logger.info("Memulai eksekusi skrip run.py...")
    parser = argparse.ArgumentParser(description="Skrip Deteksi Api & Asap menggunakan YOLO (Mode Command-Line).")
    parser.add_argument("--model", type=str, default="best.pt", help="Path menuju file model YOLO (.pt). Default: best.pt")
    parser.add_argument("--source", type=str, default="0", help="Path menuju file video, ID kamera (misalnya '0' untuk default), atau string 'camera'. Default: 0")
    parser.add_argument("--confidence", type=float, default=0.20, help="Ambang batas kepercayaan deteksi (0.0-1.0). Default: 0.20") 
    parser.add_argument("--iou", type=float, default=0.45, help="Ambang batas IoU NMS (0.0-1.0). Default: 0.45")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran gambar input model (kelipatan 32). Default: 640")
    parser.add_argument("--augment", action='store_true', help="Aktifkan Test-Time Augmentation (TTA).")
    parser.add_argument("--clahe", action='store_true', help="Aktifkan pra-pemrosesan CLAHE.")
    parser.add_argument("--cooldown", type=int, default=10, help="Cooldown notifikasi (detik). Default: 10") 
    parser.add_argument("--telegram", action='store_true', help="Aktifkan notifikasi Telegram.")
    parser.add_argument("--whatsapp", action='store_true', help="Aktifkan notifikasi WhatsApp (CallMeBot).")
    # Argumen --imgur dihilangkan karena notifier.py menangani logika Imgur secara internal
    parser.add_argument("--gemini", action='store_true', help="Aktifkan analisis Gemini AI.")
    parser.add_argument("--location", type=str, default="CLI-Feed-DetectorX", help="Nama lokasi monitoring. Default: CLI-Feed-DetectorX")
    
    args = parser.parse_args()
    logger.info(f"Argumen CLI yang diterima: {args}")

    logger.info("Memulai validasi argumen CLI...")
    if not (0.0 <= args.confidence <= 1.0):
        logger.error("Nilai --confidence harus antara 0.0 dan 1.0. Menghentikan skrip."); exit(1)
    if not (0.0 <= args.iou <= 1.0):
        logger.error("Nilai --iou (IoU threshold) harus antara 0.0 dan 1.0. Menghentikan skrip."); exit(1)
    
    model_file = Path(args.model)
    if not model_file.is_file():
        logger.error(f"File model tidak ditemukan di path: {args.model} (Path absolut: {model_file.resolve()}). Pastikan file model ada di lokasi yang benar atau berikan path yang sesuai. Menghentikan skrip.")
        exit(1)
    else:
        logger.info(f"File model '{args.model}' ditemukan.")

    logger.info("Validasi argumen CLI selesai. Memanggil process_video_source_cli...")
    process_video_source_cli(
        args.source, args.model, 
        args.confidence, args.iou, args.imgsz, args.augment,
        args.clahe, args.cooldown, 
        args.telegram, args.whatsapp, 
        args.gemini, args.location
    )
    logger.info("Fungsi process_video_source_cli telah selesai dieksekusi.")