# Driver Drowsiness Detection Demo

A professional, real-time Streamlit application for detecting driver drowsiness using computer vision and voice alerts. Built for executive demos, funding presentations, and rapid prototyping in AI safety.

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

---

## 🚗 Project Overview
This project demonstrates a robust, real-time driver drowsiness detection system using:
- **OpenCV** for webcam access and image processing
- **MediaPipe** for facial landmark detection
- **Eye Aspect Ratio (EAR)** for drowsiness detection
- **DeepFace** for robust face detection and cropping
- **pyttsx3** for continuous voice alerts
- **Streamlit** for a modern, interactive UI

The app is designed for professional presentations, with a focus on reliability, clarity, and executive appeal.

---

## 🧠 Algorithm Details
**Drowsiness is detected using the Eye Aspect Ratio (EAR):**
- **DeepFace** is used to robustly detect and crop the driver's face from the image or webcam frame, improving landmark accuracy and reliability across lighting and pose variations.
- Facial landmarks are extracted from the cropped face using MediaPipe's Face Mesh.
- The EAR is calculated for both eyes using six key landmarks per eye.
- If the EAR falls below a threshold (default: **0.25**), the driver is considered drowsy.
- The system provides both visual (red alert) and continuous voice alerts until the driver is alert again.
- The sidebar includes a **live EAR threshold slider** for real-time tuning of the drowsiness sensitivity.
- The algorithm now uses **only EAR** (not emotion) for drowsiness detection, for maximum reliability. Detected emotion is displayed in the UI for transparency, but is not used for alerting.

**Key Steps:**
1. Capture video frames from the webcam.
2. Use DeepFace to detect and crop the face region.
3. Detect facial landmarks on the cropped face.
4. Compute EAR for both eyes.
5. Trigger alerts if drowsiness is detected.

---

## 📁 Folder Structure
```
.
├── frontend.py                # Main Streamlit application (was streamlit_app.py)
├── drowsiness_realtime.py     # Standalone OpenCV/MediaPipe reference script
├── drowsiness_app.py          # Alternative Streamlit implementation
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files/folders excluded from version control
├── drowsienv/                 # (Local) Python virtual environment (not tracked)
```

---

## ⚡️ Quick Start (Local)
1. **Clone the repository:**
   ```bash
   git clone https://github.com/amioykr82/driver-drowsiness-demo.git
   cd driver-drowsiness-demo
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```bash
   streamlit run frontend.py
   ```

---

## ☁️ Deploy to Streamlit Cloud
1. **Push this repo to your GitHub account** ([amioykr82](https://github.com/amioykr82)).
2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)** and connect your repo.
3. **Set the main file** to `frontend.py` and deploy.
4. **Share your public Streamlit link** for demos and feedback.

---

## 🖥️ Usage
- **Webcam Demo:**
  - Click "Start Webcam Detection" to begin real-time monitoring.
  - Visual and voice alerts will trigger if drowsiness is detected.
  - Use the **EAR threshold slider in the sidebar** to tune the sensitivity live.
  - Click "Stop Webcam Detection" to end.
- **Image Upload:**
  - Upload a driver image to test drowsiness detection on static photos.
- **Navigation:**
  - Use the sidebar for About, Team, and Contact info.

---

## 📝 Professional Notes
- **Voice Alert:** The system uses a background thread to repeat the voice alert as long as drowsiness is detected, and stops immediately when the driver is alert.
- **No personal data is stored.**
- **Virtual environment folders** (e.g., `drowsienv/`) are excluded from version control via `.gitignore`.

---

## 👨‍💻 Author & Credits
- **Amioy Kumar** ([GitHub](https://github.com/amioykr82) | [LinkedIn](https://linkedin.com/in/amioykr))
- For demo requests, partnerships, or investment opportunities: amioy.iitd@gmail.com

---

## 📜 License
This project is for demonstration and educational purposes. For commercial use, please contact the author.
