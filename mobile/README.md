
# Basketball Form Analyzer - Mobile App

This project is a **React Native mobile application** for analyzing basketball shooting form using computer vision. It integrates **TensorFlow Lite models** and for real-time **pose detection** and **object detection**, and communicates with a backend server for post-recording analysis and storage.

---

## 🚀 Features
- Real-time **pose detection** using MoveNet Thunder.
- **Object detection** using YOLOv8n (TFLite).
- Frame-by-frame **data collection** and visualization.
- **Backend integration** for storage and phase processing.
- Dynamic **screen rendering** with keypoints, bounding boxes, and skeleton lines.

---

## 📱 Mobile App Setup

### Requirements
- Node.js (>= 14.x)
- Yarn or npm
- React Native CLI
- Android Studio (for Android builds)
- Xcode (for iOS builds)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/basketball-form-analyzer.git
cd basketball-form-analyzer

# 2. Install JS dependencies
yarn install  # or npm install

# 3. Install iOS dependencies (only for macOS/iOS builds)
cd ios && pod install && cd ..

# 4. Start Metro bundler
yarn start
````

### Run the App

```bash
# For iOS
yarn ios

# For Android
yarn android
```

---

## 🖥️ Backend Setup

The mobile app communicates with a **FastAPI** backend for pose and detection data processing.

### Requirements

* Python (>= 3.8)
* FastAPI or Flask

### Installation

```bash
# 1. Navigate to the backend folder
cd backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\\venv\\Scripts\\activate (Windows)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the backend server
uvicorn main:app --reload  # For FastAPI
```

---

## 📁 Folder Structure

```
basketball-form-analyzer/
├── mobile/                   # React Native frontend
├── backend/               # FastAPI/Flask backend
├── data_collection
├── data
├── pose_extration
├── phase_detection
├──basketball_shooting_analyzer.py
└── basketball_shooting_integrated_pipeline.py
```

---

## 💡 Key Commands

### Mobile App

```bash
# Start Metro
yarn start

# iOS build
yarn ios

# Android build
yarn android
```

### Backend

```bash
# Start backend
uvicorn main:app --reload

# Install backend dependencies
pip install -r requirements.txt
```

---

## 🔗 API Endpoints

### Backend API

* `POST /model-output`: Accepts frame data with keypoints and detections
* `POST /model-output/processed`: Triggers backend analysis pipeline
* `GET /data`: Returns processed frame/shot results
