**README.md**

# AI-Enhanced Facial Emotion Recognition in Video

## Overview

This project combines facial landmark detection and emotion recognition to analyze emotions in a video using artificial intelligence. It utilizes computer vision techniques, TensorFlow, and Mediapipe to detect facial landmarks and a pre-trained model to predict emotions. The resulting video is annotated with emotion labels for a more immersive viewing experience.

## Dependencies

Ensure you have the following dependencies installed:

- OpenCV (cv2)
- Mediapipe
- NumPy
- TensorFlow (with Keras)
- Various other Python libraries

To install the required dependencies, run the following command:

```bash
pip install opencv-python mediapipe numpy tensorflow
```

## Setting Up Virtual Environment

1. Create and activate a virtual environment:

   ```bash
   # Create a virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

2. Install dependencies in the virtual environment:

   ```bash
   pip install -r requirements.txt
   ```

3. Verify the installation:

   ```bash
   python -c "import cv2; import mediapipe; import numpy; import tensorflow"
   ```

4. When you're done, you can deactivate the virtual environment:

   ```bash
   deactivate
   ```

## Downloading Required Files

1. **FER (Facial Emotion Recognition) Model:**

   - Download the FER model files from the following link:
     [FER Model](https://github.com/priya-dwivedi/face_and_emotion_detection/tree/master/emotion_detector_models)
     
     Place the downloaded files in the project directory, and update the file path in the script:

     ```python
     fer_model = load_model("path/to/FER_model.h5")  # Replace with the actual path
     ```

2. **Emotion Recognition Retail Model:**

   - Download the Emotion Recognition Retail model files from the OpenVINO Model Zoo:
     [Emotion Recognition Retail Model](https://github.com/openvinotoolkit/openvino_models/tree/main/intel/emotions-recognition-retail-0003/FP32)

     Place the downloaded files (emotions-recognition-retail-0003.bin and emotions-recognition-retail-0003.xml) in the project directory, and update the file paths in the script:

     ```python
     emotion_model_bin = "path/to/emotions-recognition-retail-0003.bin"  # Replace with the actual path
     emotion_model_xml = "path/to/emotions-recognition-retail-0003.xml"  # Replace with the actual path
     ```

3. **Other Dependencies:**

   - Install the necessary Python libraries by running:

     ```bash
     pip install opencv-python mediapipe numpy tensorflow
     ```

   - Make sure to have OpenCV, Mediapipe, NumPy, and TensorFlow installed.


## Usage

1. Clone this repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the FER (Facial Emotion Recognition) model:

   Replace `"path/to/FER_model.h5"` with the actual path where the FER model is located.

4. Run the script:

```bash
python emotion_recognition.py
```

The script will process the input video, annotate facial landmarks, and display the video with emotion labels.

## Notes

- The script assumes the input video is located at `"nosubs.mp4"`. Replace this with the actual path of your video.
- The output video will be saved as `"nosubs-output.mp4"`.
- Press `q` to exit the application.

## Additional Information

For more details on the project and its components, refer to the code documentation and associated resources.

## Acknowledgments

- [Mediapipe](https://google.github.io/mediapipe/) for providing facial landmark detection capabilities.
- [FER Model](https://github.com/priya-dwivedi/face_and_emotion_detection/tree/master/emotion_detector_models) for emotion recognition.

Feel free to explore and modify the code to suit your specific requirements.

---

*Note: Adjust the paths, video filenames, and other configurations based on your local setup.*