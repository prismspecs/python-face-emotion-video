import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load FER model
fer_model = load_model("FER_model.h5")  # Replace with the actual path

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def get_emotion_label(emotion_scores):
    return emotions[np.argmax(emotion_scores)]


def get_emotion_labels(emotion_scores):
    # Sort emotions by score in descending order
    emotion_labels = [
        (emotion, score)
        for emotion, score in zip(emotions, emotion_scores)
        if score > 0.1
    ]
    emotion_labels.sort(key=lambda x: x[1], reverse=True)
    return "\u000A".join(
        [f"{emotion}: {score:.2%}" for emotion, score in emotion_labels]
    )


def draw_emotion_labels(frame, pos_x, pos_y, emotion_labels):
    # Get the text size
    text_size = cv2.getTextSize("test", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

    # Extract the height from the text size
    text_height = text_size[1]

    for i, line in enumerate(emotion_labels.split("\n")):
        y_offset = int(i * text_height * 1.1)
        cv2.putText(
            frame,
            line,
            (pos_x - 1, pos_y + y_offset - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        cv2.putText(
            frame,
            line,
            (pos_x, pos_y + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )


def main():
    cap = cv2.VideoCapture("nosubs.mp4")  # Replace with the actual video path

    # get the framerate of cap
    fps = cap.get(cv2.CAP_PROP_FPS)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "nosubs-output.mp4", fourcc, fps, (int(cap.get(3)), int(cap.get(4)))
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        # Check if face landmarks are detected
        if result.multi_face_landmarks:
            for facial_landmarks in result.multi_face_landmarks:
                landmarks = np.array([(pt.x, pt.y) for pt in facial_landmarks.landmark])

                # Add a check for the presence of face landmarks
                if landmarks is not None:
                    # Convert normalized coordinates to pixel coordinates
                    landmarks_pixels = (
                        landmarks * np.array([frame.shape[1], frame.shape[0]])
                    ).astype(int)

                    # Calculate convex hull for the face outline
                    hull = cv2.convexHull(landmarks_pixels)

                    # Draw face outline
                    cv2.drawContours(frame, [hull], -1, (0, 0, 255, 126), 1)

                    # get size in pixels of the bounding rectangle
                    (x, y, w, h) = cv2.boundingRect(landmarks_pixels)

                    # three levels of detail based on how large the bounding rectangle is
                    if w > 0.3 * frame.shape[1]:
                        detail_level = 3
                    elif w > 0.2 * frame.shape[1]:
                        detail_level = 2
                    else:
                        detail_level = 1

                    # Draw facial landmarks with different colors
                    for idx, (x, y) in enumerate(landmarks_pixels):
                        if idx < 17:  # Jawline
                            color = (255, 0, 0)  # Blue
                        elif idx < 22:  # Right eyebrow
                            color = (0, 255, 0)  # Green
                        elif idx < 27:  # Left eyebrow
                            color = (0, 0, 255)  # Red
                        elif idx < 36:  # Nose
                            color = (255, 255, 0)  # Yellow
                        elif idx < 48:  # Right eye
                            color = (255, 0, 255)  # Magenta
                        else:  # Left eye
                            color = (0, 255, 255)  # Cyan

                        if detail_level == 3:
                            # draw landmark idx as small text at landmark location
                            cv2.putText(
                                frame,
                                str(idx),
                                (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                color,
                                1,
                            )

                        if detail_level == 2:
                            cv2.circle(frame, (x, y), 1, color, -1)

                    # Calculate bounding rectangle
                    (x, y, w, h) = cv2.boundingRect(landmarks_pixels)

                    # draw bounding rec
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Add a check for empty or invalid bounding rectangle
                    if (
                        x >= 0
                        and y >= 0
                        and w > 0
                        and h > 0
                        and x + w <= frame.shape[1]
                        and y + h <= frame.shape[0]
                    ):
                        face_roi = frame[y : y + h, x : x + w]

                        # Add a check for empty face ROI
                        if not face_roi.size == 0:
                            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                            face_roi_gray = cv2.resize(face_roi_gray, (48, 48))
                            face_roi_gray = face_roi_gray / 255.0
                            face_roi_gray = np.expand_dims(face_roi_gray, axis=0)

                            # Predict emotion
                            emotion_scores = fer_model.predict(face_roi_gray)[0]

                            emotion_labels = get_emotion_labels(emotion_scores)

                            pos_x = int(x + w / 2)
                            pos_y = int(y + h * 0.05)

                            draw_emotion_labels(frame, pos_x, pos_y, emotion_labels)

            cv2.imshow("Facial Landmarks with Emotion", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
