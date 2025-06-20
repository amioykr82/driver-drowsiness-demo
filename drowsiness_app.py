import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import time
from PIL import Image

# --- UI Setup ---
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("Driver Drowsiness Detection")
st.markdown(
    '<div style="margin-top: 1em; font-size: 1.1em; color: #888;">'
    'This demo uses your webcam to detect drowsiness using facial landmarks.\n'
    'Please allow camera access and position your face in the frame.\n'
    '<b>.</b>'
    '</div>',
    unsafe_allow_html=True
)

# --- Voice Alert ---
def speak_alert(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

# --- Drowsiness Detection Logic ---
mp_face_mesh = mp.solutions.face_mesh
EYE_LANDMARKS = {
    'left': [33, 160, 158, 133, 153, 144],
    'right': [362, 385, 387, 263, 373, 380]
}

def eye_aspect_ratio(landmarks, eye_indices):
    # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    p = [np.array([landmarks[i][0], landmarks[i][1]]) for i in eye_indices]
    vertical1 = np.linalg.norm(p[1] - p[5])
    vertical2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

def detect_drowsiness(landmarks):
    left_ear = eye_aspect_ratio(landmarks, EYE_LANDMARKS['left'])
    right_ear = eye_aspect_ratio(landmarks, EYE_LANDMARKS['right'])
    ear = (left_ear + right_ear) / 2.0
    return ear < 0.21, ear

# --- Camera Selection and Initialization ---
def get_camera():
    for idx in [0, 1]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap, idx
        cap.release()
    return None, None

# --- State Management ---
if 'drowsy_counter' not in st.session_state:
    st.session_state['drowsy_counter'] = 0
if 'normal_counter' not in st.session_state:
    st.session_state['normal_counter'] = 0
if 'last_voice_status' not in st.session_state:
    st.session_state['last_voice_status'] = None
if 'last_state' not in st.session_state:
    st.session_state['last_state'] = 'Initializing'
if 'last_summary' not in st.session_state:
    st.session_state['last_summary'] = ''
if 'ear' not in st.session_state:
    st.session_state['ear'] = 0.0

FRAME_WINDOW = st.empty()
alert_placeholder = st.empty()
summary_placeholder = st.empty()

# --- Main Camera Loop ---
run_detection = st.button(
    'Start Drowsiness Detection', key='start_detection_btn')

if run_detection or st.session_state.get('detection_running', False):
    st.session_state['detection_running'] = True
    stop_btn = st.button('Stop Detection', key='stop_detection_btn')
    cap, cam_idx = get_camera()
    if cap is None:
        border_color = "#e53935"
        alert = (
            "Webcam not accessible. Please check your camera permissions."
        )
        troubleshooting_html = (
            '<div style="background: #fff3f3; color: #b31217; border-radius: 12px; '
            'padding: 16px 24px; margin: 10px 0 20px 0; font-size: 1em; '
            'border-left: 6px solid #e53935;">'
            'Troubleshooting:<ul>'
            '<li>Ensure no other app is using the webcam.</li>'
            '<li>Try a different browser (Chrome/Edge/Firefox).</li>'
            '<li>Check camera permissions in your OS and browser.</li>'
            '<li>Try unplugging/replugging the camera.</li>'
            '<li>Restart your computer if needed.</li>'
            '</ul></div>'
        )
        alert_html = (
            '<div style="background: #e53935; color: #fff; border-radius: 16px; '
            'padding: 24px 32px; margin: 20px 0; font-size: 1.2em; font-weight: bold; '
            'border-left: 12px solid {border_color};">{alert}</div>'
        )
        alert_html = alert_html.format(border_color=border_color, alert=alert)
        alert_placeholder.markdown(
            alert_html + troubleshooting_html,
            unsafe_allow_html=True
        )
        st.session_state['detection_running'] = False
        st.stop()
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while st.session_state.get('detection_running', False):
            ret, frame = cap.read()
            if not ret or frame is None:
                st.session_state['last_state'] = 'Camera Error'
                st.session_state['last_summary'] = 'Webcam not accessible.'
                frame = np.zeros((400, 400, 3), dtype=np.uint8)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                h, w, _ = frame.shape
                landmarks = []
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for lm in face_landmarks.landmark:
                            landmarks.append((int(lm.x * w), int(lm.y * h)))
                    is_drowsy, ear = detect_drowsiness(landmarks)
                    st.session_state['ear'] = ear
                    if is_drowsy:
                        st.session_state['drowsy_counter'] += 1
                        st.session_state['normal_counter'] = 0
                    else:
                        st.session_state['normal_counter'] += 1
                        st.session_state['drowsy_counter'] = 0
                    if st.session_state['drowsy_counter'] > 15:
                        st.session_state['last_state'] = 'Drowsy'
                        st.session_state['last_summary'] = (
                            f"Drowsiness detected! (EAR={ear:.2f})"
                        )
                    elif st.session_state['normal_counter'] > 10:
                        st.session_state['last_state'] = 'Normal'
                        st.session_state['last_summary'] = (
                            f"Driver is alert. (EAR={ear:.2f})"
                        )
                    color = (
                        (0, 255, 0)
                        if st.session_state['last_state'] == 'Normal'
                        else (0, 0, 255)
                    )
                    cv2.putText(
                        frame,
                        st.session_state['last_state'],
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        color,
                        3,
                    )
                else:
                    st.session_state['last_state'] = 'Face Not Detected'
                    st.session_state['last_summary'] = (
                        'No face detected. Please face the camera.'
                    )
            FRAME_WINDOW.image(frame, channels="RGB")
            # --- UI Alerts ---
            state = st.session_state['last_state']
            summary = st.session_state['last_summary']
            ear = st.session_state['ear']
            if state == 'Drowsy':
                border_color = "#e53935"
                alert = "Drowsiness detected! Please take a break."
                alert_placeholder.markdown(
                    f'<div style="background: linear-gradient(90deg, #e53935 0%, #b31217 100%); color: #fff; border-radius: 16px; padding: 24px 32px; margin: 20px 0; font-size: 1.5em; font-weight: bold; box-shadow: 0 4px 24px rgba(229,57,53,0.15); border-left: 12px solid {border_color};">{alert}</div>',
                    unsafe_allow_html=True
                )
                # Play voice alert if previous state was not Drowsy
                if st.session_state.get('last_voice_status') != 'Drowsy':
                    speak_alert(alert)
                st.session_state['last_voice_status'] = 'Drowsy'
            elif state == 'Normal':
                border_color = "#43a047"
                alert = "Driver is alert."
                alert_placeholder.markdown(
                    f'<div style="background: linear-gradient(90deg, #43a047 0%, #2193b0 100%); color: #fff; border-radius: 16px; padding: 24px 32px; margin: 20px 0; font-size: 1.5em; font-weight: bold; box-shadow: 0 4px 24px rgba(67,160,71,0.15); border-left: 12px solid {border_color};">{alert}</div>',
                    unsafe_allow_html=True
                )
                st.session_state['last_voice_status'] = 'Normal'
            elif state == 'Face Not Detected':
                border_color = "#888"
                alert = "No face detected. Please face the camera."
                alert_placeholder.markdown(
                    f'<div style="background: #eee; color: #222; border-radius: 16px; padding: 24px 32px; margin: 20px 0; font-size: 1.5em; font-weight: bold; border-left: 12px solid {border_color};">{alert}</div>',
                    unsafe_allow_html=True
                )
            elif state == 'Camera Error':
                border_color = "#e53935"
                alert = "Webcam not accessible. Please check your camera permissions."
                alert_placeholder.markdown(
                    f'<div style="background: #e53935; color: #fff; border-radius: 16px; padding: 24px 32px; margin: 20px 0; font-size: 1.2em; font-weight: bold; border-left: 12px solid {border_color};">{alert}</div>'
                    '<div style="background: #fff3f3; color: #b31217; border-radius: 12px; padding: 16px 24px; margin: 10px 0 20px 0; font-size: 1em; border-left: 6px solid #e53935;">'
                    'Troubleshooting:<ul>'
                    '<li>Ensure no other app is using the webcam.</li>'
                    '<li>Try a different browser (Chrome/Edge/Firefox).</li>'
                    '<li>Check camera permissions in your OS and browser.</li>'
                    '<li>Try unplugging/replugging the camera.</li>'
                    '<li>Restart your computer if needed.</li>'
                    '</ul></div>',
                    unsafe_allow_html=True
                )
            else:
                border_color = "#888"
                alert = "Initializing camera..."
                alert_placeholder.markdown(
                    f'<div style="background: #eee; color: #222; border-radius: 16px; padding: 24px 32px; margin: 20px 0; font-size: 1.5em; font-weight: bold; border-left: 12px solid {border_color};">{alert}</div>',
                    unsafe_allow_html=True
                )
            if summary:
                summary_placeholder.markdown(
                    f'<div style="background: #f7f8fa; color: #222; border-radius: 12px; padding: 18px 24px; margin: 10px 0 20px 0; font-size: 1.15em; border-left: 8px solid {border_color};">{summary}</div>',
                    unsafe_allow_html=True
                )
            # Add a small delay for UI responsiveness
            import time
            time.sleep(0.07)
            if stop_btn:
                st.session_state['detection_running'] = False
                break
        cap.release()
        st.success(f"Camera released (index {cam_idx}).")
        st.session_state['last_state'] = 'Initializing'
        st.session_state['last_summary'] = ''
        st.session_state['drowsy_counter'] = 0
        st.session_state['normal_counter'] = 0
        st.session_state['ear'] = 0.0
else:
    st.session_state['detection_running'] = False
    st.info('Click the "Start Drowsiness Detection" button above to begin.')
