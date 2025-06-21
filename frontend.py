import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from PIL import Image
import tempfile
import threading
import time
import os
from drowsiness_detection import AdvancedDrowsinessDetector
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# Set environment variables to avoid GPU conflicts
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        try:
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
        except Exception as e:
            st.error(f"Voice alert error: {e}")

    def stop(self):
        if self._active:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=1)
            self._active = False

# --- WebRTC Video Transformer for Real-time Processing ---
class DrowsinessVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = AdvancedDrowsinessDetector()
        self.ear_threshold = 0.25
        self.voice_alert_manager = VoiceAlertManager()
        self.drowsy_state = False
        self.frame_count = 0
        self.processing_interval = 5  # Process every 5 frames to reduce CPU load

    def transform(self, frame):
        self.frame_count += 1
        
        # Process every few frames to reduce CPU load
        if self.frame_count % self.processing_interval != 0:
            return frame
        
        try:
            # Convert frame to OpenCV format
            img = frame.to_ndarray(format="bgr24")
            
            # Detect drowsiness
            is_drowsy, ear, emotion, error = self.detector.detect_drowsiness(img)
            
            # Use only EAR for drowsiness detection
            is_drowsy = (ear is not None and ear < self.ear_threshold)
            
            # Handle drowsiness state changes
            previous_state = self.drowsy_state
            if is_drowsy:
                if not previous_state:
                    # Just became drowsy - start voice alert
                    self.voice_alert_manager.start("Drowsiness detected! Please take a break.")
                self.drowsy_state = True
                
                # Draw drowsiness alert on frame
                cv2.putText(img, f"DROWSY! EAR: {ear:.3f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 3)
            else:
                if previous_state:
                    # Just became alert - stop voice alert
                    self.voice_alert_manager.stop()
                self.drowsy_state = False
                
                # Draw alert status on frame
                cv2.putText(img, f"Alert EAR: {ear:.3f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add emotion info
            if emotion and emotion != 'unknown':
                cv2.putText(img, f"Emotion: {emotion}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert back to av.VideoFrame
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            # If processing fails, return original frame
            return frame

# --- Streamlit UI ---
st.title("🚗 Driver Drowsiness Detection Demo")
st.markdown("""
A professional, demo-ready Streamlit application for real-time driver drowsiness detection using computer vision and voice alerts. Built for showcasing to executives and investors.

- **Real-time webcam drowsiness detection (WebRTC + MediaPipe + DeepFace)**
- **Visual and voice alerts**
- **Modern, branded UI (Streamlit)**
- **About, Team, and Contact sections**

""")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Demo", "About", "Team", "Contact"])

# Initialize session state
if 'voice_alert_manager' not in st.session_state:
    st.session_state['voice_alert_manager'] = VoiceAlertManager()
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
    
    # Camera permission and setup info
    st.info("""
    📹 **Camera Access Required**
    
    This app needs access to your camera for real-time drowsiness detection. 
    When you click "Start Camera", your browser will ask for camera permissions.
    
    **How it works:**
    1. Click "Start Camera" below
    2. Allow camera access when prompted by your browser
    3. The app will start monitoring for drowsiness in real-time
    4. Visual and voice alerts will trigger if drowsiness is detected
    5. Use the EAR threshold slider in the sidebar to adjust sensitivity
    """)
    
    # WebRTC Configuration
    rtc_configuration = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    # Create video transformer with current threshold
    def create_video_transformer():
        transformer = DrowsinessVideoTransformer()
        transformer.ear_threshold = st.session_state['ear_threshold']
        return transformer
    
    # WebRTC Streamer
    webrtc_ctx = webrtc_streamer(
        key="drowsiness-detection",
        video_transformer_factory=create_video_transformer,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "facingMode": "user"  # Use front camera
            },
            "audio": False
        },
        async_processing=True,
    )
    
    # Status display
    if webrtc_ctx.state.playing:
        st.success("✅ Camera is active and monitoring for drowsiness!")
        st.info("💡 **Tips:** Look directly at the camera and try closing your eyes to test the drowsiness detection.")
    else:
        st.warning("⏸️ Camera is not active. Click 'Start Camera' to begin monitoring.")
    
    # Manual stop button
    if st.button("Stop Camera"):
        if webrtc_ctx.state.playing:
            webrtc_ctx.stop()
            st.session_state['voice_alert_manager'].stop()
            st.rerun()

    st.markdown("---")
    st.subheader("Or try with an uploaded image:")
    uploaded = st.file_uploader("Upload a driver image", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            frame = np.array(img.convert('RGB'))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Initialize detector if not already done
            if 'ai_detector' not in st.session_state:
                st.session_state['ai_detector'] = AdvancedDrowsinessDetector()
            
            is_drowsy, ear, emotion, error = st.session_state['ai_detector'].detect_drowsiness(frame)
            # Use only EAR for drowsiness detection
            is_drowsy = (ear is not None and ear < st.session_state['ear_threshold'])
            
            debug_placeholder = st.empty()
            debug_placeholder.info(f"EAR: {ear:.3f if ear is not None else 'N/A'} | Emotion: {emotion}")
            
            previous_state = st.session_state.get('drowsy_state', False)
            if error:
                st.info(f"Detection error: {error}")
            else:
                if is_drowsy:
                    st.warning(f"Drowsiness Detected! 🚨 (EAR={ear:.3f if ear is not None else 'N/A'})")
                    if not previous_state:
                        st.session_state['voice_alert_manager'].start("Drowsiness detected in image!")
                    st.session_state['drowsy_state'] = True
                else:
                    st.success(f"Driver looks alert. (EAR={ear:.3f if ear is not None else 'N/A'})")
                    if previous_state:
                        st.session_state['voice_alert_manager'].stop()
                    st.session_state['drowsy_state'] = False
        except Exception as e:
            st.error(f"Error processing uploaded image: {e}")

elif page == "About":
    st.header("About This Project")
    st.write("""
    This demo showcases a real-time driver drowsiness detection system using computer vision and AI. Built for executive demos and funding presentations.
    """)
    st.markdown("- **Tech:** Streamlit, WebRTC, OpenCV, MediaPipe, DeepFace, pyttsx3")
    st.markdown("- **Vision:** Safer roads through AI-powered driver monitoring.")
    st.markdown("- **Features:** Real-time camera monitoring, voice alerts, image upload analysis")

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

# Cleanup when app stops
if 'voice_alert_manager' in st.session_state:
    st.session_state['voice_alert_manager'].stop() 