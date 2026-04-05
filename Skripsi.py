import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import time
import av  # Penting untuk pengolahan video modern
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_js_eval import streamlit_js_eval

# ===============================
# 1. KONFIGURASI HALAMAN & CSS
# ===============================
st.set_page_config(page_title="Jeruk Siam AI", page_icon="🍊", layout="wide")

# CSS Original Anda + Perbaikan tampilan video
st.markdown("""
<style>
[data-testid="stSidebar"] { display: none; }
video { width: 100% !important; height: auto !important; border-radius: 12px; border: 3px solid #FFA500; }
.instruction-box {
    background-color: #262730;
    color: #FFFFFF;
    padding: 20px;
    border-radius: 10px;
    border-left: 6px solid #FFA500;
    margin-bottom: 20px;
    font-size: 16px;
    line-height: 1.6;
}
.instruction-box h4 { color: #FFA500; margin-top: 0; }
.tech-tag {
    font-size: 12px;
    color: #aaaaaa;
    margin-top: 10px;
    font-style: italic;
    border-top: 1px solid #444;
    padding-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 2. DETEKSI DEVICE
# ===============================
screen_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH')
is_mobile = screen_width is not None and screen_width < 768

# ===============================
# 3. LOAD MODEL (Cached)
# ===============================
@st.cache_resource
def load_models():
    keras_path = "model_jeruk_rgb_final.keras"
    # Menggunakan yolov8n (nano) agar ringan di server Cloud
    detector = YOLO("yolov8n.pt") 
    
    if os.path.exists(keras_path):
        classifier = tf.keras.models.load_model(keras_path)
    else:
        st.error(f"File {keras_path} tidak ditemukan di direktori.")
        classifier = None
    return detector, classifier

detector, classifier = load_models()

# ===============================
# 4. UI HEADER & PANDUAN
# ===============================
st.title("🍊 Deteksi & Klasifikasi Kualitas Jeruk Siam")

st.markdown("""
<div class="instruction-box">
<h4>Panduan Penggunaan</h4>
<ol>
<li>Tekan <b>START</b> untuk mengaktifkan kamera.</li>
<li>Dekatkan jeruk ke arah kamera (jarak 15–30 cm).</li>
<li>Pastikan pencahayaan ruangan terang agar warna kulit terdeteksi akurat.</li>
<li>Tahan posisi beberapa detik sampai hasil analisis muncul.</li>
</ol>
<div class="tech-tag">Powered by: YOLOv8 (Detection) + MobileNetV2 (Classification)</div>
</div>
""", unsafe_allow_html=True)

# ===============================
# 5. VIDEO PROCESSOR (The Engine)
# ===============================
class OrangeAnalyzer(VideoProcessorBase):
    def __init__(self):
        self.detector = detector
        self.classifier = classifier
        self.memory = {}
        self.frame_count = 0
        self.threshold = 0.5

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Analisa dilakukan setiap 5 frame untuk menghemat CPU di Cloud
        if self.frame_count % 5 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        current_time = time.time()

        # Membersihkan objek yang sudah hilang dari kamera (lebih dari 2 detik)
        self.memory = {
            k: v for k, v in self.memory.items()
            if current_time - v["last_seen"] <= 2
        }

        # Jalankan YOLO Tracking (imgsz kecil = lebih cepat)
        results = self.detector.track(
            img, 
            persist=True, 
            conf=0.4, 
            classes=[47, 49], # Apple/Orange di dataset COCO
            imgsz=320, 
            verbose=False
        )

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = box

                if obj_id not in self.memory:
                    self.memory[obj_id] = {
                        "scores": [],
                        "decision": None,
                        "last_seen": current_time
                    }

                mem = self.memory[obj_id]
                mem["last_seen"] = current_time

                # Jika belum ada keputusan final, lakukan klasifikasi MobileNetV2
                if mem["decision"] is None and self.classifier is not None:
                    crop = img[max(0,y1):y2, max(0,x1):x2]

                    if crop.size > 0:
                        # Pre-processing sesuai standar training Anda
                        crop_input = cv2.resize(crop, (224, 224))
                        crop_input = cv2.cvtColor(crop_input, cv2.COLOR_BGR2RGB)
                        crop_input = crop_input.astype("float32") / 255.0
                        crop_input = np.expand_dims(crop_input, axis=0)

                        pred = self.classifier.predict(crop_input, verbose=0)
                        score = float(pred[0][0])
                        mem["scores"].append(score)

                        # Menampilkan progres analisa sementara
                        progress = int((len(mem["scores"]) / 15) * 100)
                        label_temp = "MANIS" if score > self.threshold else "ASAM"
                        color_temp = (0, 255, 0) if score > self.threshold else (0, 0, 255)
                        
                        cv2.putText(img, f"{label_temp} ({progress}%)", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_temp, 2)

                        # Jika sudah terkumpul 15 sampel, ambil keputusan rata-rata
                        if len(mem["scores"]) >= 15:
                            avg_score = np.mean(mem["scores"])
                            mem["decision"] = "MANIS" if avg_score > 0.5 else "ASAM"

                # Jika keputusan sudah final, tampilkan label tetap & Confidence
                elif mem["decision"] is not None:
                    final_label = mem["decision"]
                    color = (0, 255, 0) if final_label == "MANIS" else (0, 0, 255)
                    
                    # Hitung persentase keyakinan
                    avg_score = np.mean(mem["scores"])
                    conf_val = avg_score if final_label == "MANIS" else (1 - avg_score)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, final_label, (x1, y1 - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(img, f"Conf: {conf_val*100:.1f}%", (x1, y2 + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===============================
# 6. WEBRTC STREAMER LAUNCHER
# ===============================
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="jeruk-app",
    video_processor_factory=OrangeAnalyzer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "facingMode": "environment" if is_mobile else "user",
            "width": {"ideal": 640},
            "frameRate": {"ideal": 20}
        },
        "audio": False
    },
    async_processing=True,
)
