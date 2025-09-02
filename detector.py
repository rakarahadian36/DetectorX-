# detector.py
from ultralytics import YOLO
import cv2
import logging
import numpy as np 

logger = logging.getLogger(__name__)

class YoloDetector:
    """
    Kelas untuk melakukan deteksi objek menggunakan model YOLO (Ultralytics).
    """
    def __init__(self, model_path='best.pt'):
        """
        Inisialisasi detektor.
        Args:
            model_path (str): Path menuju file model .pt YOLO.
        """
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names 
            if not self.class_names or not isinstance(self.class_names, dict) or not all(isinstance(k, int) for k in self.class_names.keys()):
                logger.warning(f"Nama kelas dari model.names tidak valid atau bukan dictionary yang diharapkan (menerima: {self.class_names}).")
                # !!! PENTING: SESUAIKAN ID KELAS DAN NAMA DI BAWAH INI JIKA PERLU !!!
                self.class_names = {0: 'fire', 1: 'smoke'} 
                logger.warning(f"Menggunakan nama kelas default: {self.class_names}. PASTIKAN INI SESUAI DENGAN MODEL ANDA DAN URUTAN KELASNYA!")
            else:
                logger.info(f"Nama kelas berhasil dimuat dari model: {self.class_names}")
            logger.info(f"Model YOLO '{model_path}' berhasil dimuat.")
        except Exception as e:
            logger.error(f"Gagal memuat model YOLO dari '{model_path}': {e}", exc_info=True)
            self.model = None
            self.class_names = {} 

    def detect(self, frame_input, confidence_threshold=0.25, iou_threshold=0.45, imgsz=640, augment=False):
        """
        Melakukan deteksi objek pada frame gambar.
        Args:
            frame_input (np.ndarray): Frame gambar input (bisa BGR, BGRA, atau Grayscale).
            confidence_threshold (float): Ambang batas kepercayaan minimum untuk deteksi.
            iou_threshold (float): Ambang batas IoU untuk Non-Maximum Suppression.
            imgsz (int): Ukuran gambar input untuk model.
            augment (bool): Apakah menggunakan Test-Time Augmentation.
        Returns:
            tuple: (detected_objects_list, annotated_frame)
                   detected_objects_list (list): List berisi dictionary info deteksi yang relevan (fire/smoke).
                   annotated_frame (np.ndarray): Frame BGR dengan anotasi (bounding box, label).
        """
        if self.model is None:
            logger.warning("Model YOLO tidak dimuat, proses deteksi dilewati.")
            return [], frame_input 

        logger.debug(f"Frame input awal ke detect(): shape={frame_input.shape}, dtype={frame_input.dtype}")

        processed_frame = frame_input 
        if frame_input.ndim == 2: 
            logger.info("Input frame adalah grayscale. Mengkonversi ke BGR (3 channel)...")
            processed_frame = cv2.cvtColor(frame_input, cv2.COLOR_GRAY2BGR)
        elif frame_input.ndim == 3:
            if frame_input.shape[2] == 4: 
                logger.info("Input frame memiliki 4 channel (misalnya BGRA). Mengkonversi ke BGR (3 channel)...")
                processed_frame = cv2.cvtColor(frame_input, cv2.COLOR_BGRA2BGR)
            elif frame_input.shape[2] == 3: 
                logger.debug("Input frame sudah 3 channel (diasumsikan BGR/RGB). Tidak ada konversi channel yang dilakukan.")
            else:
                logger.error(f"Input frame 3 dimensi memiliki jumlah channel tidak terduga: {frame_input.shape[2]}. Diharapkan 3 atau 4. Mengembalikan frame asli.")
                return [], frame_input
        else:
            logger.error(f"Format dimensi input frame tidak didukung (ndim={frame_input.ndim}). Diharapkan 2 atau 3. Mengembalikan frame asli.")
            return [], frame_input

        if not (processed_frame.ndim == 3 and processed_frame.shape[2] == 3):
            logger.error(f"Setelah upaya konversi, frame masih bukan 3 channel BGR. Shape akhir: {processed_frame.shape}. Menghentikan deteksi untuk frame ini.")
            return [], frame_input 
        
        logger.debug(f"Frame FINAL yang akan diinput ke model.predict: shape={processed_frame.shape}, dtype={processed_frame.dtype}")

        results_list = self.model.predict(
            source=processed_frame, 
            conf=confidence_threshold, 
            iou=iou_threshold,
            imgsz=imgsz,
            augment=augment,
            verbose=False 
        )

        detected_objects_list = []
        annotated_frame = processed_frame.copy() 

        if not results_list:
            logger.info("Tidak ada hasil deteksi (results_list kosong) dari model predict.")
            return [], annotated_frame 
        
        try:
            results = results_list[0] 
            annotated_frame_with_plots = results.plot() 
            if annotated_frame_with_plots is not None and annotated_frame_with_plots.shape == annotated_frame.shape:
                 annotated_frame = annotated_frame_with_plots 
            elif annotated_frame_with_plots is not None:
                 logger.warning(f"Hasil dari results.plot() memiliki shape berbeda ({annotated_frame_with_plots.shape}) dari frame input ({annotated_frame.shape}). Menggunakan frame sebelum plot.")
            else:
                logger.warning("Hasil dari results.plot() adalah None. Menggunakan frame sebelum plot.")

            if logger.isEnabledFor(logging.DEBUG): # Hanya log detail jika level DEBUG aktif
                logger.debug(f"--- Hasil Mentah dari Model Predict (conf_model={confidence_threshold}) ---")
                if len(results.boxes) == 0:
                    logger.debug("Tidak ada kotak deteksi sama sekali dari model.predict().")
                for i, box in enumerate(results.boxes):
                    class_id_raw = int(box.cls[0])
                    conf_raw = float(box.conf[0])
                    label_raw = self.class_names.get(class_id_raw, f"UnknownID_{class_id_raw}")
                    logger.debug(f"  Mentah {i+1}: Label='{label_raw}' (ID:{class_id_raw}), Confidence={conf_raw:.4f}, Bbox={box.xyxy[0].cpu().numpy().astype(int).tolist()}")
                logger.debug("--- Akhir Hasil Mentah ---")
            
            for box in results.boxes: # Iterasi lagi untuk memfilter fire/smoke
                class_id_raw = int(box.cls[0])
                conf_raw = float(box.conf[0])
                label_raw = self.class_names.get(class_id_raw, f"UnknownID_{class_id_raw}")
                if label_raw.lower() in ['fire', 'smoke']: # Filter utama di sini
                    detected_objects_list.append({
                        'label': label_raw,
                        'confidence': conf_raw,
                        'bbox': box.xyxy[0].cpu().numpy().astype(int).tolist()
                    })
            
            if not detected_objects_list and len(results.boxes) > 0:
                logger.info(f"Objek terdeteksi oleh model ({len(results.boxes)}), tapi setelah difilter label, tidak ada 'fire' atau 'smoke'.")
            elif not detected_objects_list:
                 logger.info("Tidak ada objek 'fire' atau 'smoke' yang terdeteksi (atau tidak melewati ambang conf model).")
            else:
                logger.info(f"Berhasil memfilter {len(detected_objects_list)} objek 'fire'/'smoke' dari hasil model.")

        except Exception as e:
            logger.error(f"Error saat memproses hasil deteksi YOLO: {e}", exc_info=True)

        return detected_objects_list, annotated_frame