import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from streamlit_js_eval import streamlit_js_eval


# ===============================
# 1. KONFIGURASI HALAMAN
# ===============================
st.set_page_config(page_title="Jeruk Siam AI", page_icon="🍊", layout="wide")

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
# 3. LOAD MODEL
# ===============================
@st.cache_resource
def load_models():
    keras_path = "model_jeruk_rgb_final.keras"

    detector = YOLO("yolov8n.pt")

    if os.path.exists(keras_path):
        classifier = tf.keras.models.load_model(keras_path)
    else:
        st.error(f"File {keras_path} tidak ditemukan.")
        classifier = None

    return detector, classifier


detector, classifier = load_models()


# ===============================
# 4. UI
# ===============================
st.title("🍊 Deteksi & Klasifikasi Kualitas Jeruk Siam")

st.markdown("""
<div class="instruction-box">
<h4>Panduan Penggunaan</h4>
<ol>
<li>Tekan START untuk mengaktifkan kamera.</li>
<li>Dekatkan jeruk (15–30 cm).</li>
<li>Pastikan pencahayaan terang.</li>
<li>Tahan posisi sampai hasil terkunci.</li>
</ol>
<div class="tech-tag">YOLOv8 + MobileNetV2</div>
</div>
""", unsafe_allow_html=True)


# ===============================
# 5. VIDEO PROCESSOR
# ===============================
class OrangeAnalyzer(VideoTransformerBase):
    def __init__(self):
        self.detector = detector
        self.classifier = classifier
        self.memory = {}
        self.frame_count = 0
        self.threshold = 0.5

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % 7 != 0:
            return img

        current_time = time.time()

        # Hapus objek lama
        self.memory = {
            k: v for k, v in self.memory.items()
            if current_time - v["last_seen"] <= 2
        }

        results = self.detector.track(
            img,
            persist=True,
            conf=0.5,
            classes=[47, 49],
            imgsz=320,
            verbose=False
        )

        if not results or results[0].boxes.id is None:
            return img

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = box

            label_text = "Menganalisa..."
            color = (255, 255, 255)

            if obj_id not in self.memory:
                self.memory[obj_id] = {
                    "scores": [],
                    "decision": None,
                    "last_seen": current_time
                }

            mem = self.memory[obj_id]
            mem["last_seen"] = current_time

            if mem["decision"] is None and self.classifier is not None:
                crop = img[y1:y2, x1:x2]

                if crop.size > 0:
                    crop = cv2.resize(crop, (224, 224))
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop = crop.astype("float32") / 255.0
                    crop = np.expand_dims(crop, axis=0)

                    pred = self.classifier.predict(crop, verbose=0)
                    score = float(pred[0][0])
                    mem["scores"].append(score)

                    temp_label = "MANIS" if score > self.threshold else "ASAM"
                    color = (0, 255, 0) if score > self.threshold else (0, 0, 255)
                    progress = int((len(mem["scores"]) / 15) * 100)
                    
                    # Output: MANIS (20%) atau ASAM (20%)
                    label_text = f"{temp_label} ({progress}%)"

                    if len(mem["scores"]) >= 15:
                        avg = np.mean(mem["scores"])
                        mem["decision"] = "MANIS" if avg > 0.12 else "ASAM"

            elif mem["decision"] is not None:
                label_text = mem["decision"]
                color = (0, 255, 0) if label_text == "MANIS" else (0, 0, 255)
                
                # --- TAMBAHAN: Teks Confidence Score ---
                # Menghitung % keyakinan dari rata-rata score yang tersimpan
                avg_score = np.mean(mem["scores"])
                conf_val = avg_score if label_text == "MANIS" else (1 - avg_score)
                conf_text = f"Conf: {conf_val*100:.1f}%"
                
                # Menampilkan di bawah kiri kotak (y2 + 25)
                cv2.putText(
                    img, 
                    conf_text, 
                    (x1, y2 + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    color, 
                    2
                )

            # Gambar Box & Label Atas (Tetap sesuai code asli Anda)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                img,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        return img


# ===============================
# 6. WEBRTC CONFIG
# ===============================
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="jeruk-app",
    video_transformer_factory=OrangeAnalyzer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"facingMode": "environment", "width": {"ideal": 640}},
        "audio": False
    },
    async_processing=True,
)
