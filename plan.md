# Project Plan: AI-Enhanced Facial Emotion Recognition in Video

## Overview

This project processes video files to detect faces, extract facial landmarks, and recognize emotions using AI models. The output is a video annotated with emotion labels for each detected face, providing an immersive and informative viewing experience.

## Tech Stack

- **Programming Language:** Python 3.x
- **Core Libraries:**
  - OpenCV (`cv2`) – video processing and display
  - Mediapipe – facial landmark detection
  - NumPy – numerical operations
  - TensorFlow (with Keras) – loading and running emotion recognition models
- **Pre-trained Models:**
  - FER (Facial Emotion Recognition) model (`FER_Model.h5`)
  - Optionally, OpenVINO Emotion Recognition Retail model

## Directory Structure

```
/python-face-emotion-video/
│
├── face.py                    # Main script for video processing and emotion recognition
├── FER_Model.h5               # Pre-trained FER model file
├── model_v6_23.hdf5           # (Optional) Another model file, if used
├── input.mp4                  # Example input video
├── input-old.mp4              # (Optional) Another input video
├── output-with-audio.mp4      # Example output video
├── requirements.txt           # Python dependencies
├── readme.md                  # Project documentation and setup instructions
└── plan.md                    # (This file) Project plan and navigation guide
```

## Main Components

- **Video Input/Output:** Reads an input video file, processes each frame, and writes the annotated output video.
- **Face Detection & Landmark Extraction:** Uses Mediapipe to detect faces and extract facial landmarks in each frame.
- **Emotion Recognition:** Loads a pre-trained Keras/TensorFlow model (e.g., `FER_Model.h5`) to predict emotions from facial features.
- **Annotation:** Draws facial landmarks and overlays emotion labels on the video frames.
- **Script Entry Point:** The main logic is likely in `face.py`, which orchestrates the above steps.

## Setup & Usage

1. **Install Dependencies:**
   - Use `pip install -r requirements.txt` to install required Python packages.

2. **Download Models:**
   - Place the FER model (`FER_Model.h5`) in the project directory.
   - (Optional) Download and place OpenVINO model files if using that path.

3. **Run the Script:**
   - `python face.py` (or `python emotion_recognition.py` if that's the main script)
   - The script processes `input.mp4` (or another specified video) and outputs an annotated video.

## Key Files

- `face.py`: Main script for running the pipeline.
- `FER_Model.h5`: Pre-trained emotion recognition model.
- `requirements.txt`: List of required Python packages.
- `readme.md`: Detailed setup, usage, and troubleshooting instructions.

## Notes for LLMs & Developers

- Paths to model files and videos may need to be updated in the script.
- The code assumes certain filenames for input/output; these can be parameterized for flexibility.
- The project is modular: face detection, landmark extraction, and emotion recognition are distinct steps.
- The code is designed for batch processing of videos, not real-time webcam input (unless modified).