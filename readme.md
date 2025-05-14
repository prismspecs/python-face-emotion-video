# AI-Enhanced Facial Emotion Recognition in Video

## Overview

This project analyzes emotions in video files using facial landmark detection and emotion recognition powered by AI. It uses OpenCV for video processing, Mediapipe for facial landmark detection, and TensorFlow/Keras for emotion recognition with a pre-trained model. The output is a video annotated with emotion labels for each detected face.

## Features
- Detects faces and facial landmarks in video frames
- Recognizes emotions using a pre-trained deep learning model
- Annotates video frames with emotion labels
- Outputs an annotated video file

## Tech Stack
- Python 3.x
- OpenCV (`cv2`)
- Mediapipe
- NumPy
- TensorFlow (with Keras)

## Directory Structure
```
face.py                  # Main script for video processing and emotion recognition
FER_Model.h5             # Pre-trained FER model file (download separately)
requirements.txt         # Python dependencies
readme.md                # Project documentation
plan.md                  # Project plan and navigation guide
input.mp4                # Example input video (replace with your own)
output-with-audio.mp4    # Example output video
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd python-face-emotion-video
```

### 2. Create and Activate a Virtual Environment
#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
#### On Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the FER Model
- Download the FER model file (`FER_Model.h5`) from [here](https://github.com/priya-dwivedi/face_and_emotion_detection/tree/master/emotion_detector_models).
- Place `FER_Model.h5` in the project directory.
- Update the model path in the script if necessary:
  ```python
  fer_model = load_model("FER_Model.h5")
  ```

### 5. Prepare Your Input Video
- Place your input video in the project directory (e.g., `input.mp4`).
- Update the input/output filenames in the script if needed.

### 6. Run the Script
```bash
python face.py
```
- The script will process the input video, annotate facial landmarks, and display/save the video with emotion labels.

### 7. Deactivate the Virtual Environment (when done)
```bash
deactivate
```

## Notes
- The script assumes the input video is named `input.mp4` and the output will be saved as `output-with-audio.mp4`. You can change these in the script.
- Make sure the FER model file is present in the project directory.
- Press `q` to exit the video display window during processing.

## Troubleshooting
- If you encounter missing package errors, ensure your virtual environment is activated and dependencies are installed.
- For model loading errors, verify the path to `FER_Model.h5` is correct.

## Acknowledgments
- [Mediapipe](https://google.github.io/mediapipe/) for facial landmark detection
- [FER Model](https://github.com/priya-dwivedi/face_and_emotion_detection/tree/master/emotion_detector_models) for emotion recognition
