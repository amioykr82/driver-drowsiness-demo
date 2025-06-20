import cv2
import numpy as np
from deepface import DeepFace
import torch
import mediapipe as mp

class AdvancedDrowsinessDetector:
    def __init__(self, detector_backend='opencv'):
        self.detector_backend = detector_backend
        self.mp_face_mesh = mp.solutions.face_mesh
        self.EYE_LANDMARKS = {
            'left': [33, 160, 158, 133, 153, 144],
            'right': [362, 385, 387, 263, 373, 380]
        }

    def detect_face(self, image):
        faces = DeepFace.extract_faces(img_path=image, detector_backend=self.detector_backend, enforce_detection=False)
        if len(faces) > 0:
            face_dict = faces[0]
            # Debug: return type and content if detection fails
            if isinstance(face_dict, dict):
                if "facial_area" in face_dict:
                    area = face_dict["facial_area"]
                    x = area.get("x", 0)
                    y = area.get("y", 0)
                    w = area.get("w", image.shape[1])
                    h = area.get("h", image.shape[0])
                    return (x, y, w, h)
                else:
                    # fallback: use the whole image as face
                    h, w = image.shape[:2]
                    return (0, 0, w, h)
            elif isinstance(face_dict, np.ndarray):
                h, w = face_dict.shape[:2]
                return (0, 0, w, h)
            else:
                # fallback: use the whole image as face, but also return debug info
                h, w = image.shape[:2]
                raise Exception(f"Unknown face_dict type: {type(face_dict)}, value: {repr(face_dict)}")
        return None

    def crop_face(self, image, face_box):
        x, y, w, h = face_box
        return image[y:y+h, x:x+w]

    def detect_drowsiness(self, image):
        try:
            face_box = self.detect_face(image)
        except Exception as e:
            return False, None, None, f'Debug: {str(e)}'
        if face_box is None:
            return False, None, None, 'No face detected'
        face_img = self.crop_face(image, face_box)
        try:
            preds = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            # Handle both list and dict output
            if isinstance(preds, list):
                if len(preds) > 0 and isinstance(preds[0], dict) and 'dominant_emotion' in preds[0]:
                    emotion = preds[0]['dominant_emotion']
                else:
                    return False, None, None, f'DeepFace analyze returned list but no dominant_emotion: {repr(preds)}'
            elif isinstance(preds, dict):
                emotion = preds.get('dominant_emotion', None)
            else:
                return False, None, None, f'DeepFace analyze returned unknown type: {type(preds)}'
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            ) as face_mesh:
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        h, w, _ = face_img.shape
                        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                        ear = self.eye_aspect_ratio(landmarks, self.EYE_LANDMARKS['left'])
                        ear_r = self.eye_aspect_ratio(landmarks, self.EYE_LANDMARKS['right'])
                        avg_ear = (ear + ear_r) / 2.0
                        is_drowsy = (avg_ear < 0.21)
                        return is_drowsy, avg_ear, emotion, None
            is_drowsy = (avg_ear < 0.21)
            return is_drowsy, avg_ear, emotion, None
        except Exception as e:
            return False, None, None, f'DeepFace error: {str(e)}'

    @staticmethod
    def eye_aspect_ratio(landmarks, eye_indices):
        p = [np.array([landmarks[i][0], landmarks[i][1]]) for i in eye_indices]
        vertical1 = np.linalg.norm(p[1] - p[5])
        vertical2 = np.linalg.norm(p[2] - p[4])
        horizontal = np.linalg.norm(p[0] - p[3])
        return (vertical1 + vertical2) / (2.0 * horizontal)

# Example usage:
# detector = AdvancedDrowsinessDetector()
# is_drowsy, ear, emotion, error = detector.detect_drowsiness(image)
# if error:
#     print('Detection error:', error)
# elif is_drowsy:
#     print('Drowsiness detected!')
# else:
#     print('Driver is alert.') 