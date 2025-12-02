# 🤟 Indian Sign Language (ISL) Real-time Recognition System

A comprehensive machine learning application for recognizing Indian Sign Language gestures in real-time using computer vision, deep learning, and natural language processing.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Model Training](#model-training)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This system enables real-time recognition of Indian Sign Language gestures through a camera feed, providing instant translation to multiple Indian languages with audio output. The application uses MediaPipe for pose estimation, scikit-learn for classification, and offers a user-friendly GUI built with Tkinter.

### Key Capabilities
- **Real-time Detection**: Recognizes ISL gestures from webcam feed at 30 FPS
- **Multi-language Support**: Translates to Tamil, Telugu, Kannada, Malayalam, Hindi, and English
- **Text-to-Speech**: Audio output in all supported languages
- **High Accuracy**: Configurable confidence threshold with visual feedback
- **Flexible Training**: Support for custom gesture datasets

## ✨ Features

### 🎥 Frame Collection & Processing
- **Camera Integration**: Multi-camera support with index selection
- **Frame Capture**: Real-time frame capture at 30 FPS
- **Preprocessing**: Automatic frame flipping, resizing, and RGB conversion
- **Pose Detection**: MediaPipe Holistic for full-body landmark extraction

### 🤖 Training Pipeline
- **Feature Extraction**: 258-point full body or 194-point upper body keypoints
- **Data Processing**: StandardScaler normalization
- **Model**: RandomForestClassifier with probability outputs
- **Persistence**: Model, scaler, and action labels saved as `.pkl` files

### 🔍 Detection & Recognition
- **Real-time Inference**: Continuous gesture recognition
- **Confidence Filtering**: Adjustable threshold (0.1 - 1.0)
- **Auto-detection**: Configurable interval (1-10 seconds)
- **Manual Mode**: On-demand prediction with button click

### 🌐 Translation & Audio
- **Multi-language Translation**: Google Translator API integration
- **Text-to-Speech**: gTTS with pygame audio playback
- **Language Options**: 6 Indian languages supported
- **Audio Queue**: Non-blocking threaded audio processing

### 🖥️ User Interface
- **Modern Dark Theme**: Professional UI with color-coded feedback
- **Video Display**: Live camera feed with overlay annotations
- **Statistics Panel**: Real-time FPS, confidence, and prediction count
- **History Tracking**: Last 50 predictions with timestamps
- **Separate File Loading**: Individual buttons for model, scaler, and actions

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam/Camera
- Windows/Linux/macOS

### Step 1: Clone Repository
### Step 2: Create Virtual Environment (Recommended)
### Step 3: Install Dependencies
### Step 4: Install Additional Packages


## 🚀 Usage

### 1️⃣ Frame Collection
Collect training data by capturing video frames:

**Options:**
- `--action`: Name of the gesture (e.g., "hello", "thank_you")
- `--sequences`: Number of video sequences to record
- `--frames`: Frames per sequence

### 2️⃣ Processing Frames
Extract keypoints from collected frames:

**Output:** Numpy arrays with MediaPipe keypoints for each frame

### 3️⃣ Model Training
Train the RandomForest classifier:


**Generated Files:**
- `isl_model.pkl` - Trained classifier
- `isl_scaler.pkl` - Feature scaler
- `isl_actions.pkl` - Action labels

### 4️⃣ Real-time Detection
Launch the GUI application:


**GUI Workflow:**
1. Click **📁 Model**, **📊 Scaler**, **🏷️ Actions** to load files
2. Select camera index and click **Connect**
3. Choose output language
4. Adjust confidence threshold and detection interval
5. Click **▶ Start Detection**
6. Perform ISL gestures in front of camera
7. View predictions with translations and audio


## 🛠️ Technologies

### Core Libraries
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Core language |
| **OpenCV** | 4.5+ | Video capture and processing |
| **MediaPipe** | 0.10.11 | Pose and hand landmark detection |
| **scikit-learn** | 1.7+ | Machine learning (RandomForest) |
| **NumPy** | 1.21+ | Numerical computing |

### GUI & Audio
| Technology | Purpose |
|-----------|---------|
| **Tkinter** | Desktop GUI framework |
| **PIL/Pillow** | Image processing for GUI |
| **pygame** | Audio playback |
| **gTTS** | Text-to-speech generation |

### Translation
| Technology | Purpose |
|-----------|---------|
| **deep-translator** | Multi-language translation |
| **Google Translate API** | Translation backend |

## 🎓 Model Training

### Dataset Requirements
- **Minimum**: 30 sequences per gesture
- **Frames per sequence**: 30 frames
- **Recommended gestures**: 20-50 unique signs
- **Recording environment**: Good lighting, plain background

### Training Parameters

FEATURES = 258 # Full body: 258, Upper body: 194
CONFIDENCE_THRESHOLD = 0.25
N_ESTIMATORS = 100 # RandomForest trees
TEST_SPLIT = 0.2
RANDOM_STATE = 42


### Performance Metrics
- **Accuracy**: Typically 85-95% on test set
- **Inference Speed**: ~30 FPS on CPU
- **Model Size**: ~5-10 MB per 50 gestures

## ⚙️ Configuration

### Language Codes

-LANGUAGES = {
'Tamil': 'ta',
'Telugu': 'te',
'Kannada': 'kn',
'Malayalam': 'ml',
'English': 'en',
'Hindi': 'hi'
}

### Camera Settings

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
CAMERA_INDEX = 0 # Change for external cameras

### MediaPipe Configuration

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5


## 🐛 Troubleshooting

### Common Issues

**Problem**: `InconsistentVersionWarning` on model load
Solution: Retrain model or suppress warning
pip install scikit-learn==1.7.1

**Problem**: `SymbolDatabase GetPrototype Error`
solution: Fix protobuf version
pip install protobuf==3.20.3

**Problem**: Translation not working
Solution: Install deep-translator
pip install deep-translator

**Problem**: Camera not detected
Solution: Try different camera indices (0, 1, 2)

## 📊 Performance Optimization

### Speed Improvements
- Use upper body only (194 features) for faster inference
- Reduce detection interval for less frequent predictions
- Lower camera resolution if needed
- Use GPU for MediaPipe if available

### Accuracy Improvements
- Collect more training data (50+ sequences per gesture)
- Use consistent lighting and background
- Increase confidence threshold to reduce false positives
- Add data augmentation during training

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

pip install -r requirements-dev.txt
pre-commit install
pytest tests/


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** by Google for pose estimation framework
- **scikit-learn** for machine learning tools
- **OpenCV** community for computer vision library
- Indian Sign Language research community
- All contributors and testers

---

**⭐ If you find this project helpful, please give it a star!**

Made with ❤️ for the Indian Sign Language community
