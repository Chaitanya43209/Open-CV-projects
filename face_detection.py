# Import Libraries
import cv2
import time
import mediapipe as mp
import collections

# Initialize Mediapipe Holistic Model and Drawing Utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ✅ Use DirectShow backend for better camera compatibility on Windows
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if camera opened successfully
if not capture.isOpened():
    print("❌ Could not open camera. Try changing index (0 → 1 or 2).")
    exit()

# Initialize time variables
previous_time = 0
fps_values = collections.deque(maxlen=10)  # for smooth FPS

# Use context manager for holistic model
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic_model:

    while True:
        ret, frame = capture.read()
        if not ret or frame is None:
            print("⚠️ Unable to capture frame. Check your camera connection.")
            break

        # Resize frame for performance
        frame = cv2.resize(frame, (640, 480))

        # Convert BGR to RGB for Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic_model.process(image)
        image.flags.writeable = True

        # Convert RGB back to BGR for OpenCV display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw facial landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
            )

        # Draw right hand landmarks
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        # Draw left hand landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        # Calculate FPS
        current_time = time.time()
        if previous_time != 0:
            fps = 1 / (current_time - previous_time)
            fps_values.append(fps)
            avg_fps = sum(fps_values) / len(fps_values)
        else:
            avg_fps = 0
        previous_time = current_time

        # Display FPS on the frame
        cv2.putText(image, f"{int(avg_fps)} FPS", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the result
        cv2.imshow("Facial and Hand Landmarks", image)

        # Exit when 'ESC' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Clean up
capture.release()
cv2.destroyAllWindows()