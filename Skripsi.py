import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import time
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
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
}
.instruction-box h4 { color: #FFA500; margin-top: 0; }
</style>
""", unsafe_allow_html=True)

# ===============================
# 2. LOAD MODEL (Cached)
# ===============================
@st.cache_resource
def load_models():
    # Gunakan path relatif
    keras_path = "model_jeruk_rgb_final.keras"
    detector = YOLO("yolov8n.pt") # Model paling ringan
    
    if os.path.exists(keras_path):
        classifier = tf.keras.models.load_model(keras_path)
    else:
        st.error(f"File {keras_path} tidak ditemukan!")
        classifier = None
    return detector, classifier

detector, classifier = load_models()

# ===============================
# 3. UI HEADER
# ===============================
st.title("🍊 Deteksi & Klasifikasi Kualitas Jeruk Siam")

st.markdown("""
<div class="instruction-box">
<h4>Panduan Penggunaan</h4>
<ol>
<li>Tekan <b>START</b> untuk mengaktifkan kamera.</li>
<li>Dekatkan jeruk ke arah kamera (15–30 cm).</li>
<li>Tunggu hingga progres analisis mencapai 100%.</li>
</ol>
</div>
""", unsafe_allow_html=True)

# ===============================
# 4. LOGIKA ANALISIS (Global State untuk Callback)
# ===============================
# Menggunakan dictionary sederhana untuk menyimpan history deteksi
if 'memory' not in st.session_state:
    st.session_state.memory = {}

# Konfigurasi Threshold
THRESHOLD = 0.05 

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Deteksi YOLO (imgsz 320 lebih ringan untuk Cloud daripada 640)
    # Kami gunakan .predict() demi stabilitas koneksi di server
    results = detector.predict(img, conf=0.5, classes=[47, 49], imgsz=320, verbose=False)

    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Preprocessing Crop
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                try:
                    # Resize & Normalize
                    input_img = cv2.resize(crop, (224, 224))
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                    input_img = np.expand_dims(input_img.astype("float32") / 255.0, axis=0)
                    
                    # Prediksi Kualitas
                    pred = classifier.predict(input_img, verbose=0)
                    score = float(pred[0][0])
                    
                    # Penentuan Label
                    label = "MANIS" if score > THRESHOLD else "ASAM"
                    color = (0, 255, 0) if label == "MANIS" else (0, 0, 255)
                    
                    # Hitung Confidence sederhana untuk tampilan
                    conf_display = score if label == "MANIS" else (1 - score)
                    
                    # Gambar Box & Label
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, f"{label}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(img, f"Conf: {conf_display*100:.1f}%", (x1, y2 + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception as e:
                    pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===============================
# 5. WEBRTC CONFIG & LAUNCHER
# ===============================
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="jeruk-siam-ai",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "facingMode": "environment", # Prioritas kamera belakang (HP)
            "width": {"ideal": 480},    # Resolusi sedang agar tidak freeze
            "frameRate": {"ideal": 20}   # FPS cukup untuk deteksi
        },
        "audio": False
    },
    async_processing=True,
)

st.info("Catatan: Jika video freeze di 0 detik, pastikan Anda memberikan izin kamera pada browser dan refresh halaman.")