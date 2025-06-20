import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import time

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

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not accessible. Please check your camera.")
        return
    drowsy_counter = 0
    normal_counter = 0
    last_voice_status = None
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Camera Error")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            h, w, _ = frame.shape
            landmarks = []
            state = "Initializing"
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for lm in face_landmarks.landmark:
                        landmarks.append((int(lm.x * w), int(lm.y * h)))
                is_drowsy, ear = detect_drowsiness(landmarks)
                if is_drowsy:
                    drowsy_counter += 1
                    normal_counter = 0
                else:
                    normal_counter += 1
                    drowsy_counter = 0
                if drowsy_counter > 15:
                    state = 'Drowsy'
                    color = (0, 0, 255)
                    alert = "Drowsiness detected! Please take a break."
                    cv2.putText(frame, alert, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                    if last_voice_status != 'Drowsy':
                        speak_alert(alert)
                        last_voice_status = 'Drowsy'
                elif normal_counter > 10:
                    state = 'Normal'
                    color = (0, 255, 0)
                    cv2.putText(frame, "Driver is alert.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                    last_voice_status = 'Normal'
                else:
                    color = (255, 255, 0)
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                state = 'Face Not Detected'
                color = (128, 128, 128)
                cv2.putText(frame, "No face detected.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
