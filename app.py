# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image as PILImage 
import tempfile
import os
import time
from datetime import datetime
import logging
import gc 

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from detector import YoloDetector 
except ImportError:
    st.error("❌ Gagal mengimpor YoloDetector dari detector.py. Pastikan file dan nama kelas sudah benar.")
    st.stop()

from notifier import send_telegram_notification, send_whatsapp_notification
from gemini_analyzer import analyze_image_with_gemini, GEMINI_API_KEY 
from utils import apply_clahe_enhancement, image_bytes_to_rgb_numpy

MODEL_PATH_DEFAULT = 'best.pt' 
MODEL_PATH = os.getenv('MODEL_PATH', MODEL_PATH_DEFAULT) 
HEADER_IMAGE_PATH = "fireman.jpg" 


if 'detector' not in st.session_state: st.session_state.detector = None
if 'model_loaded_successfully' not in st.session_state: st.session_state.model_loaded_successfully = False
if 'last_notification_time' not in st.session_state: st.session_state.last_notification_time = {} 
if 'processing_active' not in st.session_state: st.session_state.processing_active = False 
if 'current_input_source_name' not in st.session_state: st.session_state.current_input_source_name = "Sumber Tidak Diketahui" # Untuk source_info
if 'gemini_available' not in st.session_state: st.session_state.gemini_available = bool(GEMINI_API_KEY) 
if 'whatsapp_configured' not in st.session_state: 
    st.session_state.whatsapp_configured = bool(os.getenv('CALLMEBOT_API_KEY') and os.getenv('RECEIVER_WHATSAPP_NUMBER'))

def initialize_detector(model_path_to_load):
    try:
        st.session_state.detector = YoloDetector(model_path=model_path_to_load)
        if st.session_state.detector.model: 
            st.session_state.model_loaded_successfully = True
            logger.info(f"Detector berhasil diinisialisasi dengan model: {model_path_to_load}")
            st.sidebar.success(f"✅ Model '{os.path.basename(model_path_to_load)}' dimuat!")
            # DIUBAH: Menampilkan hanya kelas target notifikasi ('fire', 'smoke') di sidebar
            alert_classes = ['fire', 'smoke']
            all_model_classes = st.session_state.detector.class_names.values()
            display_classes = [cls for cls in all_model_classes if cls.lower() in alert_classes]
            st.sidebar.caption(f"Kelas Target Notifikasi: {', '.join(display_classes)}")
        else:
            st.session_state.model_loaded_successfully = False
            st.sidebar.error(f"❌ Gagal memuat model dari '{model_path_to_load}'.")
            logger.error(f"Gagal memuat model dari '{model_path_to_load}' ke YoloDetector.")
    except Exception as e:
        st.session_state.model_loaded_successfully = False
        st.sidebar.error(f"❌ Error inisialisasi detector: {e}")
        logger.error(f"Exception saat inisialisasi YoloDetector: {e}", exc_info=True)

def can_send_notification(detection_type, cooldown_seconds):
    current_time = time.time()
    last_time = st.session_state.last_notification_time.get(detection_type, 0)
    return current_time - last_time > cooldown_seconds

def update_notification_time(detection_type):
    st.session_state.last_notification_time[detection_type] = time.time()

def attempt_remove_temp_file(file_path, max_retries=3, delay=0.1):
    if not file_path or not os.path.exists(file_path):
        logger.debug(f"Penghapusan file temporer dilewati: path tidak valid atau file tidak ada ({file_path})")
        return
    retries = 0
    while retries < max_retries:
        try:
            os.remove(file_path)
            logger.info(f"File temporer '{file_path}' berhasil dihapus pada percobaan ke-{retries+1}.")
            return 
        except PermissionError as pe:
            retries += 1
            logger.warning(f"Percobaan ke-{retries} gagal menghapus '{file_path}': {pe}. Menunggu {delay} detik.")
            if retries < max_retries: time.sleep(delay) 
            else: logger.error(f"Gagal menghapus file temporer '{file_path}' setelah {max_retries} percobaan: {pe}.")
        except Exception as e: 
            logger.error(f"Error tak terduga saat hapus '{file_path}': {e}", exc_info=True)
            return 
    
def process_frame_and_notify(frame_bgr_original_for_detection, 
                             frame_placeholder, 
                             conf_thresh, iou_thresh, imgsz, augment_tta,
                             use_clahe, notif_cooldown, 
                             enable_telegram, enable_whatsapp,
                             detection_log_area):
    if not st.session_state.model_loaded_successfully or st.session_state.detector is None:
        frame_placeholder.warning("⚠️ Model deteksi belum dimuat. Tidak dapat memproses.")
        return []

    frame_to_detect = frame_bgr_original_for_detection.copy() 
    if use_clahe: 
        with st.spinner("⚙️ Menerapkan enhancement CLAHE..."):
            frame_rgb_for_clahe = cv2.cvtColor(frame_to_detect, cv2.COLOR_BGR2RGB)
            enhanced_frame_rgb = apply_clahe_enhancement(frame_rgb_for_clahe)
            if enhanced_frame_rgb is not None: frame_to_detect = cv2.cvtColor(enhanced_frame_rgb, cv2.COLOR_RGB2BGR)
            else: logger.warning("Proses CLAHE gagal, menggunakan frame asli untuk deteksi.")
    
    detected_objects, annotated_frame_bgr_output = st.session_state.detector.detect(
        frame_to_detect, 
        confidence_threshold=conf_thresh, 
        iou_threshold=iou_thresh, 
        imgsz=imgsz, 
        augment=augment_tta
    )
    
    annotated_frame_rgb_display = cv2.cvtColor(annotated_frame_bgr_output, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(annotated_frame_rgb_display, caption="🖼️ Hasil Deteksi Visual", channels="RGB", use_container_width=True)

    detection_log_area.empty() 
    if not detected_objects: 
        detection_log_area.info("ℹ️ Tidak ada objek yang terdeteksi pada frame ini (setelah filter).")

    detected_labels_in_frame = [] 
    for det in detected_objects: 
        label, confidence, bbox = det['label'].lower(), det['confidence'], det['bbox']
        detected_labels_in_frame.append(label)
        
        # --- DIUBAH: Logika Tampilan dan Notifikasi Berdasarkan Kelas ---
        
        # 1. Logika Tampilan di Antarmuka
        if label in ['fire', 'smoke']:
            detection_log_area.warning(f"🚨 **TERDETEKSI:** `{label.capitalize()}` (Kepercayaan: {confidence:.2f}) pada bbox: `{bbox}`")
        else: # Untuk 'neutral' dan kelas lainnya
            detection_log_area.info(f"⚪️ **INFO:** Terdeteksi objek `{label.capitalize()}` (Kepercayaan: {confidence:.2f}) pada bbox: `{bbox}`")

        # 2. Logika Notifikasi (HANYA untuk kelas berbahaya)
        if label in ['fire', 'smoke'] and can_send_notification(label, notif_cooldown):
            timestamp_dt = datetime.now()
            timestamp_str = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
            
            temp_annotated_img_path = None 
            fp_temp_annotated_obj = None   
            pil_annotated_for_saving = None 
            gemini_analysis_result = None 
            
            try:
                pil_annotated_for_saving = PILImage.fromarray(cv2.cvtColor(annotated_frame_bgr_output, cv2.COLOR_BGR2RGB))
                fp_temp_annotated_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", prefix="det_annotated_")
                pil_annotated_for_saving.save(fp_temp_annotated_obj, format="JPEG")
                temp_annotated_img_path = fp_temp_annotated_obj.name 
                fp_temp_annotated_obj.close(); fp_temp_annotated_obj = None 
                logger.info(f"Frame ANOTASI disimpan ke file temporer: {temp_annotated_img_path}")
                
                if temp_annotated_img_path and st.session_state.gemini_available and st.session_state.analyze_with_gemini: 
                    with st.spinner(f"🧠 Menganalisis deteksi {label} dengan Gemini AI..."):
                        
                        gemini_prompt = (f"Gambar ini menampilkan deteksi '{label}' dari sumber '{st.session_state.current_input_source_name}' yang berlokasi di '{st.session_state.location_name}'. "
                                         "Berikan analisis singkat mengenai potensi bahaya berdasarkan apa yang terlihat pada gambar, "
                                         "kemungkinan penyebab (jika bisa disimpulkan dari visual), dan langkah-langkah keamanan dasar yang harus segera diambil. "
                                         "Fokus pada respons cepat dan tindakan preventif. Buat dalam format narasi atau poin singkat yang mudah dipahami.")

                        gemini_analysis_result = analyze_image_with_gemini(temp_annotated_img_path, gemini_prompt) 
                    
                    with detection_log_area.expander(f"🔬 Analisis & Saran dari Gemini AI untuk {label.capitalize()}", expanded=True):
                        st.markdown(gemini_analysis_result if gemini_analysis_result else "Tidak ada analisis yang diterima dari Gemini AI.")
                
                message_details_for_notif = {
                    'detection_type': label.capitalize(),
                    'location': st.session_state.location_name,
                    'source_info': st.session_state.current_input_source_name,
                    'confidence': confidence,
                    'timestamp': timestamp_str,
                    'gemini_analysis': gemini_analysis_result 
                }

                if enable_telegram:
                    with st.spinner(f"📲 Mengirim notifikasi Telegram untuk deteksi {label}..."):
                        send_telegram_notification(message_details_for_notif, image_path_annotated=temp_annotated_img_path)
                
                if enable_whatsapp and st.session_state.whatsapp_configured:
                    with st.spinner(f"📱 Mengirim notifikasi WhatsApp untuk deteksi {label}..."):
                        send_whatsapp_notification(message_details_for_notif, image_path_annotated=temp_annotated_img_path)
                
                update_notification_time(label) 
            
            except Exception as e_notify_main:
                detection_log_area.error(f"❌ Terjadi error saat proses notifikasi atau analisis Gemini: {e_notify_main}")
                logger.error(f"Error dalam loop notifikasi/analisis Gemini utama: {e_notify_main}", exc_info=True)
            
            finally:
                if hasattr(pil_annotated_for_saving, 'close'): 
                    try: pil_annotated_for_saving.close() 
                    except: pass
                if fp_temp_annotated_obj is not None : 
                    try: fp_temp_annotated_obj.close()
                    except: pass
                gc.collect() 
                if temp_annotated_img_path: 
                    attempt_remove_temp_file(temp_annotated_img_path) 
        elif label in ['fire', 'smoke']: # Kondisi ini hanya untuk menampilkan pesan cooldown jika notifikasi belum bisa dikirim
            detection_log_area.info(f"⏳ Notifikasi untuk deteksi {label} masih dalam masa cooldown.")
            
    return list(set(detected_labels_in_frame)) 

st.set_page_config(page_title="🔥💨 DetectorX - Sistem Deteksi Cerdas", layout="wide", initial_sidebar_state="expanded")
col_header1, col_header2 = st.columns([1, 6]) 
with col_header1:
    if os.path.exists(HEADER_IMAGE_PATH): st.image(HEADER_IMAGE_PATH, width=100) 
    else: st.caption(f"Logo '{HEADER_IMAGE_PATH}' tidak ditemukan.") 
with col_header2: st.title("🔥 DetectorX - Sistem Deteksi Api & Asap Cerdas 💨")
st.markdown("Selamat datang! Aplikasi ini menggunakan AI untuk mendeteksi api & asap secara *real-time*.")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Konfigurasi Sistem")
    st.session_state.location_name = st.text_input("📍 Nama/ID Lokasi", value="Area Produksi")
    if not st.session_state.detector and not st.session_state.model_loaded_successfully:
        if os.path.exists(MODEL_PATH): initialize_detector(MODEL_PATH)
        else: st.error(f"❌ Model '{MODEL_PATH}' tidak ditemukan.")
    
    with st.expander("🔧 Parameter Deteksi", expanded=False):
        conf_thresh_slider = st.slider("🎯 Kepercayaan", 0.05, 1.0, 0.15, 0.05, help="Ambang keyakinan deteksi.") 
        iou_thresh_slider = st.slider("🔗 IoU (NMS)", 0.1,1.0,0.45,0.05, help="Kontrol tumpang tindih kotak deteksi.")
        imgsz_slider = st.select_slider("🖼️ Ukuran Input", [320,416,512,640,720,1024],640, help="Ukuran gambar input model.")
        augment_tta_checkbox = st.checkbox("✨ Augmentasi (TTA)", False, help="Test-Time Augmentation.")
    with st.expander("✨ Pra-pemrosesan", expanded=False):
        use_clahe_checkbox = st.checkbox("🔆 Aktifkan CLAHE", False, help="Tingkatkan kontras gambar.")
    st.markdown("---")
    st.header("🔔 Notifikasi & AI")
    with st.expander("📢 Notifikasi", expanded=True):
        notif_cooldown_slider = st.number_input("⏱️ Cooldown (detik)",10,3600,10,10, help="Jeda antar notifikasi untuk jenis yang sama.") 
        st.session_state.enable_telegram_notif = st.checkbox("📲 Notif Telegram", True)
        if st.session_state.whatsapp_configured: st.session_state.enable_whatsapp_notif = st.checkbox("📱 Notif WhatsApp", True) 
        else: st.info("ℹ️ Notif WhatsApp belum dikonfigurasi."); st.session_state.enable_whatsapp_notif = False
    
    if st.session_state.gemini_available:
        with st.expander("🧠 Analisis Gemini AI", expanded=True):
            st.session_state.analyze_with_gemini = st.checkbox("🤖 Analisis Gemini", True, help="Jika aktif, setiap deteksi akan dianalisis oleh Gemini AI dan hasilnya disertakan dalam notifikasi utama.")
    else: 
        st.info("ℹ️ Gemini AI tidak tersedia.")
        st.session_state.analyze_with_gemini = False

    st.markdown(f"---<br><small>🕒 {datetime.now(tz=datetime.now().astimezone().tzinfo).strftime('%d %b %Y, %H:%M:%S %Z')}</small><br><small>Versi Aplikasi 2.3.4</small>", unsafe_allow_html=True)

main_col, log_col = st.columns([3, 2])
with main_col:
    st.subheader("🎬 Sumber Input & Hasil Visual")
    input_source_selection = st.selectbox("Pilih Sumber:", ("Unggah Gambar 🖼️", "Unggah Video 🎞️", "Kamera Live 📸"), label_visibility="collapsed")
    
    if input_source_selection == "Unggah Gambar 🖼️":
        st.session_state.current_input_source_name = "Unggahan Gambar"
    elif input_source_selection == "Unggah Video 🎞️":
        st.session_state.current_input_source_name = "Unggahan Video"
    elif input_source_selection == "Kamera Live 📸":
        st.session_state.current_input_source_name = "Kamera Live"
        
    frame_placeholder = st.empty() 
with log_col:
    st.subheader("📊 Log Deteksi & Analisis")
    detection_log_area = st.container(height=550, border=True) 

if not st.session_state.model_loaded_successfully: st.error("🔴 Model deteksi tidak termuat! Periksa konfigurasi.")
else:
    common_args_for_processing = (
        conf_thresh_slider, iou_thresh_slider, imgsz_slider, augment_tta_checkbox,
        use_clahe_checkbox, notif_cooldown_slider, 
        st.session_state.enable_telegram_notif, st.session_state.enable_whatsapp_notif,
        detection_log_area
    )
    if input_source_selection == "Unggah Gambar 🖼️":
        st.session_state.processing_active = False
        uploaded_file_img = st.file_uploader("Pilih file gambar:", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if uploaded_file_img:
            st.session_state.current_input_source_name = f"Unggahan Gambar: {uploaded_file_img.name}" 
            try:
                img_bytes_data = uploaded_file_img.getvalue()
                img_rgb_numpy_data = image_bytes_to_rgb_numpy(img_bytes_data)
                if img_rgb_numpy_data is not None:
                    frame_placeholder.image(img_rgb_numpy_data, caption="🖼️ Gambar Asli", use_container_width=True) 
                    if st.button("🔍 Deteksi Gambar", type="primary", use_container_width=True):
                        with st.spinner("⏳ Mendeteksi objek pada gambar..."):
                            frame_bgr_for_detection = cv2.cvtColor(img_rgb_numpy_data, cv2.COLOR_RGB2BGR)
                            process_frame_and_notify(
                                frame_bgr_for_detection, 
                                frame_placeholder, 
                                *common_args_for_processing
                            )
                            detection_log_area.success("✅ Deteksi pada gambar telah selesai.")
                else: st.error("❌ Gagal memproses file gambar yang diunggah.")
            except Exception as e_img_upload_main:
                st.error(f"❌ Terjadi error saat memproses unggahan gambar: {e_img_upload_main}")
                logger.error(f"Error pada pemrosesan unggah gambar utama: {e_img_upload_main}", exc_info=True)

    elif input_source_selection in ["Unggah Video 🎞️", "Kamera Live 📸"]:
        is_video_file_mode = input_source_selection == "Unggah Video 🎞️"
        temp_video_file_path = None 
        if is_video_file_mode:
            uploaded_video_stream = st.file_uploader("Pilih file video (MP4, AVI, MOV, MKV):", type=["mp4","avi","mov","mkv"], label_visibility="collapsed")
            if uploaded_video_stream is not None:
                st.session_state.current_input_source_name = f"Unggahan Video: {uploaded_video_stream.name}" 
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video_stream.name)[1]) as tmp_video_obj:
                    tmp_video_obj.write(uploaded_video_stream.getvalue())
                    temp_video_file_path = tmp_video_obj.name
                st.video(temp_video_file_path) 
        else:
            st.session_state.current_input_source_name = "Kamera Live"

        btn_cols_streaming = st.columns(2)
        start_btn_label = "▶️ Mulai Video" if is_video_file_mode else "▶️ Mulai Kamera"
        stop_btn_label = "⏹️ Hentikan Video" if is_video_file_mode else "⏹️ Hentikan Kamera"
        start_btn_key = f"start_btn_{'video' if is_video_file_mode else 'cam'}"
        stop_btn_key = f"stop_btn_{'video' if is_video_file_mode else 'cam'}"

        if btn_cols_streaming[0].button(start_btn_label, key=start_btn_key, use_container_width=True, type="primary", disabled=st.session_state.processing_active): 
            if is_video_file_mode and not temp_video_file_path: 
                st.warning("⚠️ Harap unggah file video terlebih dahulu.")
            else: st.session_state.processing_active = True 
        if btn_cols_streaming[1].button(stop_btn_label, key=stop_btn_key, use_container_width=True, disabled=not st.session_state.processing_active): 
            st.session_state.processing_active = False; detection_log_area.info("ℹ️ Pemrosesan dihentikan oleh pengguna.")

        if st.session_state.processing_active:
            video_capture_source = temp_video_file_path if is_video_file_mode and temp_video_file_path else 0
            if is_video_file_mode and not temp_video_file_path : st.session_state.processing_active = False 
            if st.session_state.processing_active: 
                cv_video_capture = cv2.VideoCapture(video_capture_source)
                if not cv_video_capture.isOpened(): 
                    st.error(f"❌ Gagal buka sumber input: {'File Video' if is_video_file_mode else 'Kamera Web'}."); st.session_state.processing_active = False 
                else:
                    detection_log_area.info(f"⏳ Memulai pemrosesan {'video' if is_video_file_mode else 'kamera live'}...")
                    while cv_video_capture.isOpened() and st.session_state.processing_active:
                        ret_val, frame_bgr_from_stream = cv_video_capture.read() 
                        if not ret_val: 
                            detection_log_area.success(f"✅ Pemrosesan {'video selesai.' if is_video_file_mode else 'stream kamera berakhir.'}"); 
                            st.session_state.processing_active = False; break 
                        
                        process_frame_and_notify(
                            frame_bgr_from_stream, 
                            frame_placeholder, 
                            *common_args_for_processing
                        )
                        time.sleep(0.01) 
                    cv_video_capture.release() 
                    if temp_video_file_path and os.path.exists(temp_video_file_path):
                        attempt_remove_temp_file(temp_video_file_path, max_retries=5, delay=0.2)
                    if not st.session_state.processing_active and 'ret_val' in locals() and ret_val : detection_log_area.info("ℹ️ Pemrosesan dihentikan.")
                    elif st.session_state.processing_active and 'ret_val' in locals() and not ret_val: 
                         st.session_state.processing_active = False
                         detection_log_area.success(f"✅ {'Video selesai.' if is_video_file_mode else 'Stream kamera berakhir.'}")

