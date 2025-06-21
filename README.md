# Driver Drowsiness Detection Demo

A professional, real-time Streamlit application for detecting driver drowsiness using computer vision and voice alerts. Built for executive demos, funding presentations, and rapid prototyping in AI safety.

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

---

## 🚗 Project Overview
This project demonstrates a robust, real-time driver drowsiness detection system using:
- **WebRTC** for secure, browser-based camera access
- **OpenCV** for image processing
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
1. Capture video frames from the webcam via WebRTC.
2. Use DeepFace to detect and crop the face region.
3. Detect facial landmarks on the cropped face.
4. Compute EAR for both eyes.
5. Trigger alerts if drowsiness is detected.

---

## 📁 Folder Structure
```
.
├── frontend.py                # Main Streamlit application with WebRTC
├── drowsiness_detection.py    # Core drowsiness detection logic
├── test_setup.py             # Setup verification script
├── run_app.py                # Safe startup script
├── requirements.txt          # Python dependencies
├── packages.txt              # System dependencies for Streamlit Cloud
├── .gitignore               # Files/folders excluded from version control
├── drowsienv/               # (Local) Python virtual environment (not tracked)
```

---

## ⚡️ Quick Start (Local)

### Method 1: Safe Startup (Recommended)
1. **Clone the repository:**
   ```bash
   git clone https://github.com/amioykr82/driver-drowsiness-demo.git
   cd driver-drowsiness-demo
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the safe startup script:**
   ```bash
   python run_app.py
   ```

### Method 2: Direct Streamlit
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app:**
   ```bash
   streamlit run frontend.py
   ```

---

## ☁️ Deploy to Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account

### Deployment Steps
1. **Push this repo to your GitHub account:**
   ```bash
   git add .
   git commit -m "Add WebRTC camera support for Streamlit Cloud"
   git push origin main
   ```

2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)** and:
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set the main file path to: `frontend.py`
   - Click "Deploy"

3. **Your app will be available at:** `https://your-app-name.streamlit.app`

### Camera Access on Streamlit Cloud
- The app uses **WebRTC** for secure, browser-based camera access
- Users will be prompted to allow camera permissions (like Zoom/Teams)
- Camera feed is processed in real-time and **never stored or recorded**
- Works on desktop and mobile browsers

---

## 🖥️ Usage

### Real-time Camera Monitoring
1. **Click "Start Camera"** - Your browser will ask for camera permissions
2. **Allow camera access** when prompted
3. **Look directly at the camera** - The app will start monitoring
4. **Visual and voice alerts** will trigger if drowsiness is detected
5. **Use the EAR threshold slider** in the sidebar to tune sensitivity
6. **Click "Stop Camera"** to end monitoring

### Image Upload Analysis
- **Upload a driver image** to test drowsiness detection on static photos
- Supports JPG, JPEG, and PNG formats
- Results are displayed immediately

### Navigation
- Use the sidebar for About, Team, and Contact info

---

## 🔧 Troubleshooting

### Camera Access Issues
- **Browser permissions:** Ensure you've allowed camera access when prompted
- **HTTPS required:** Camera access only works over HTTPS (automatic on Streamlit Cloud)
- **Browser compatibility:** Works best on Chrome, Firefox, Safari, Edge
- **Mobile devices:** Camera access works on mobile browsers

### Segmentation Fault Issues
If you encounter segmentation faults or CUDA-related errors locally:

1. **Use the safe startup script:**
   ```bash
   python run_app.py
   ```

2. **Set environment variables manually:**
   ```bash
   export CUDA_VISIBLE_DEVICES=-1
   export TF_CPP_MIN_LOG_LEVEL=2
   streamlit run frontend.py
   ```

3. **Test your setup:**
   ```bash
   python test_setup.py
   ```

4. **Common fixes:**
   - **GPU conflicts:** The app is configured to use CPU-only TensorFlow to avoid CUDA conflicts
   - **Memory issues:** Close other applications using GPU resources
   - **Dependency conflicts:** Use the exact versions in `requirements.txt`

### Voice Alert Issues
- **Browser limitations:** Voice alerts may not work in all browsers due to security restrictions
- **Local testing:** Voice alerts work best when running locally
- **Streamlit Cloud:** Voice alerts may be limited due to browser security policies

---

## 📝 Professional Notes
- **Privacy:** No personal data is stored. Camera feed is processed in real-time and discarded.
- **Security:** Uses WebRTC for secure, encrypted camera communication.
- **Performance:** Optimized for real-time processing with frame rate control.
- **Compatibility:** Works across different devices and browsers.
- **CPU-only mode:** Configured to avoid GPU conflicts and ensure compatibility across different systems.

---

## 👨‍💻 Author & Credits
- **Amioy Kumar** ([GitHub](https://github.com/amioykr82) | [LinkedIn](https://linkedin.com/in/amioykr))
- For demo requests, partnerships, or investment opportunities: amioy.iitd@gmail.com

---

## 📜 License
This project is for demonstration and educational purposes. For commercial use, please contact the author.
