import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from PIL import Image
import tempfile
import threading
import time
from drowsiness_detection import AdvancedDrowsinessDetector
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Branding and Config ---
st.set_page_config(page_title="Driver Drowsiness Detection Demo", page_icon="🚗", layout="wide")

# --- Logo Placeholder ---
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Driving_icon.svg/1200px-Driving_icon.svg.png", width=80)

# --- Voice Alert Function ---
class VoiceAlertManager:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._active = False

    def start(self, text):
        if self._active:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, args=(text,), daemon=True)
        self._thread.start()
        self._active = True

    def _run(self, text):
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        while not self._stop_event.is_set():
            engine.say(text)
            engine.runAndWait()
            # Wait a bit before repeating
            for _ in range(10):
                if self._stop_event.is_set():
                    break
                time.sleep(0.2)

    def stop(self):
        if self._active:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=1)
            self._active = False

# --- Streamlit UI ---
st.title("🚗 Driver Drowsiness Detection Demo")
st.markdown("""
A professional, demo-ready Streamlit application for real-time driver drowsiness detection using computer vision and voice alerts. Built for showcasing to executives and investors..

- **Real-time webcam drowsiness detection (OpenCV + MediaPipe + DeepFace)**
- **Visual and voice alerts**
- **Modern, branded UI (Streamlit)**
- **About, Team, and Contact sections**

""")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Demo", "About", "Team", "Contact"])

if 'voice_alert_manager' not in st.session_state:
    st.session_state['voice_alert_manager'] = VoiceAlertManager()
if 'ai_detector' not in st.session_state:
    st.session_state['ai_detector'] = AdvancedDrowsinessDetector()
if 'ear_threshold' not in st.session_state:
    st.session_state['ear_threshold'] = 0.25

# Add EAR threshold slider to sidebar
st.sidebar.markdown('---')
st.sidebar.subheader('EAR Threshold Tuning')
st.session_state['ear_threshold'] = st.sidebar.slider(
    'Drowsiness EAR threshold', min_value=0.10, max_value=0.35, value=st.session_state['ear_threshold'], step=0.01
)

if page == "Demo":
    st.header("Live Drowsiness Detection")
    st.write("Click 'Start' to begin webcam detection.")
    if 'run_demo' not in st.session_state:
        st.session_state['run_demo'] = False
    if 'drowsy_state' not in st.session_state:
        st.session_state['drowsy_state'] = False
    start = st.button("Start Webcam Detection")
    stop = st.button("Stop Webcam Detection")
    alert_placeholder = st.empty()
    frame_placeholder = st.empty()
    debug_placeholder = st.empty()

    if start:
        st.session_state['run_demo'] = True
    if stop:
        st.session_state['run_demo'] = False
        st.session_state['voice_alert_manager'].stop()
        st.session_state['drowsy_state'] = False

    if st.session_state['run_demo']:
        cap = cv2.VideoCapture(0)
        st.info("Press 'Stop Webcam Detection' to end the demo.")
        while cap.isOpened() and st.session_state['run_demo']:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam not detected.")
                break
            is_drowsy, ear, emotion, error = st.session_state['ai_detector'].detect_drowsiness(frame)
            # Use only EAR for drowsiness detection
            is_drowsy = (ear is not None and ear < st.session_state['ear_threshold'])
            debug_placeholder.info(f"EAR: {ear if ear is not None else 'N/A'} | Emotion: {emotion}")
            if error:
                alert_placeholder.info(f"Detection error: {error}")
                st.session_state['voice_alert_manager'].stop()
                st.session_state['drowsy_state'] = False
            else:
                previous_state = st.session_state.get('drowsy_state', False)
                if is_drowsy:
                    alert_placeholder.warning(f"Drowsiness Detected! 🚨 (EAR={ear if ear is not None else 'N/A'})")
                    # Start voice alert if just transitioned to drowsy
                    if not previous_state:
                        st.session_state['voice_alert_manager'].start("Drowsiness detected! Please take a break.")
                    st.session_state['drowsy_state'] = True
                else:
                    alert_placeholder.info(f"You look alert. ✅ (EAR={ear if ear is not None else 'N/A'})")
                    if previous_state:
                        st.session_state['voice_alert_manager'].stop()
                    st.session_state['drowsy_state'] = False
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            time.sleep(0.05)
        cap.release()
        frame_placeholder.empty()
        alert_placeholder.empty()
        debug_placeholder.empty()
        st.session_state['voice_alert_manager'].stop()
        st.session_state['drowsy_state'] = False

    st.markdown("---")
    st.subheader("Or try with an uploaded image:")
    uploaded = st.file_uploader("Upload a driver image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        frame = np.array(img.convert('RGB'))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        is_drowsy, ear, emotion, error = st.session_state['ai_detector'].detect_drowsiness(frame)
        # Use only EAR for drowsiness detection
        is_drowsy = (ear is not None and ear < st.session_state['ear_threshold'])
        debug_placeholder = st.empty()
        debug_placeholder.info(f"EAR: {ear if ear is not None else 'N/A'} | Emotion: {emotion}")
        previous_state = st.session_state.get('drowsy_state', False)
        if error:
            st.info(f"Detection error: {error}")
        else:
            if is_drowsy:
                st.warning(f"Drowsiness Detected! 🚨 (EAR={ear if ear is not None else 'N/A'})")
                if not previous_state:
                    st.session_state['voice_alert_manager'].start("Drowsiness detected in image!")
                st.session_state['drowsy_state'] = True
            else:
                st.success(f"Driver looks alert. (EAR={ear if ear is not None else 'N/A'})")
                if previous_state:
                    st.session_state['voice_alert_manager'].stop()
                st.session_state['drowsy_state'] = False

elif page == "About":
    st.header("About This Project")
    st.write("""
    This demo showcases a real-time driver drowsiness detection system using computer vision and AI. Built for executive demos and funding presentations.
    """)
    st.markdown("- **Tech:** Streamlit, OpenCV, MediaPipe, DeepFace, pyttsx3")
    st.markdown("- **Vision:** Safer roads through AI-powered driver monitoring.")

elif page == "Team":
    st.header("Meet the Team")
    st.write("""
    - **Amioy Kumar** – Senior Product Manager Founder & AI Engineer
    - **Collaborator** – Computer Vision Specialist
    - **Advisor** – Industry Expert
    """)

elif page == "Contact":
    st.header("Contact & Demo Requests")
    st.write("For demo requests, partnerships, or investment opportunities:")
    st.markdown("**Email:** amioy.iitd@gmail.com")
    st.markdown("**LinkedIn:** [Your LinkedIn](https://www.linkedin.com/in/amioykr/)")

# --- Drowsiness Detector Instance ---
if 'ai_detector' not in st.session_state:
    st.session_state['ai_detector'] = AdvancedDrowsinessDetector()

def process_drowsiness(img):
    # Run your drowsiness detection logic here
    is_drowsy, ear, emotion, error = st.session_state['ai_detector'].detect_drowsiness(img)
    # Draw result on frame
    h, w, _ = img.shape
    color = (0, 0, 255) if is_drowsy else (0, 255, 0)
    label = f"Drowsy! EAR={ear:.2f}" if is_drowsy else f"Alert EAR={ear:.2f}"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        pass
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process_drowsiness(img)
        return img

# --- WebRTC Streamer ---
webrtc_streamer(
    key="drowsiness-detection",
    video_transformer_factory=VideoTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.sidebar.title("About")
st.sidebar.info(
    "This app uses computer vision to detect drowsiness. "
    "Your webcam feed is processed in real-time and is not stored or recorded."
) 