import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from streamlit_js_eval import streamlit_js_eval

# ===============================
# 1. CONFIG
# ===============================
st.set_page_config(page_title="Jeruk Siam AI", page_icon="🍊", layout="wide")

screen_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH')
is_mobile = screen_width is not None and screen_width < 768

# ===============================
# 2. LOAD MODEL
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
# 3. VIDEO PROCESSOR
# ===============================
class OrangeAnalyzer:
    def __init__(self):
        self.detector = detector
        self.classifier = classifier
        self.frame_count = 0
        self.memory = {"scores": [], "decision": None}
        self.threshold = 0.5

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # FRAME SKIP
        if self.frame_count % 5 != 0:
            return frame

        # YOLO DETECTION (LEBIH RINGAN)
        results = self.detector.predict(
            img,
            conf=0.3,
            imgsz=320,
            verbose=False
        )

        if not results or len(results[0].boxes) == 0:
            return frame

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        for box in boxes:
            x1, y1, x2, y2 = box

            label_text = "Menganalisa..."
            color = (255, 255, 255)

            # ===============================
            # LIMIT CLASSIFICATION (MAX 5x)
            # ===============================
            if self.memory["decision"] is None and self.classifier is not None and len(self.memory["scores"]) < 5:
                crop = img[y1:y2, x1:x2]

                if crop.size > 0:
                    crop = cv2.resize(crop, (224, 224))
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop = crop.astype("float32") / 255.0
                    crop = np.expand_dims(crop, axis=0)

                    pred = self.classifier.predict(crop, verbose=0)
                    score = float(pred[0][0])
                    self.memory["scores"].append(score)

                    temp_label = "MANIS" if score > self.threshold else "ASAM"
                    color = (0, 255, 0) if score > self.threshold else (0, 0, 255)

                    progress = int((len(self.memory["scores"]) / 5) * 100)
                    label_text = f"{temp_label} ({progress}%)"

                    # FINAL DECISION
                    if len(self.memory["scores"]) >= 5:
                        avg = np.mean(self.memory["scores"])
                        self.memory["decision"] = "MANIS" if avg > 0.12 else "ASAM"

            elif self.memory["decision"] is not None:
                label_text = self.memory["decision"]
                color = (0, 255, 0) if label_text == "MANIS" else (0, 0, 255)

                avg_score = np.mean(self.memory["scores"])
                conf_val = avg_score if label_text == "MANIS" else (1 - avg_score)
                conf_text = f"Conf: {conf_val*100:.1f}%"

                cv2.putText(
                    img,
                    conf_text,
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            # DRAW BOX
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

        return frame.from_ndarray(img, format="bgr24")


# ===============================
# 4. WEBRTC
# ===============================
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="jeruk-app",
    video_processor_factory=OrangeAnalyzer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"facingMode": "environment", "width": {"ideal": 480}},
        "audio": False
    },
    async_processing=True,
)