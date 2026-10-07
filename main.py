import cv2
import mediapipe as mp
import threading
import sys
import winsound
import time
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout


# Media Pipe
mp_pose = mp.solutions.pose
posture = mp_pose.Pose()
# Drawing
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# Capture
cap = cv2.VideoCapture(0)
# Threshold
space_threshold = 190
time_threshold = 5
threshold_start = None


class CoordinatesWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coordinates Window")
        self.setGeometry(1000, 300, 500, 400)
        
        self.layout = QVBoxLayout()
        self.label = QLabel("...")
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

    def update_text(self, text):
        self.label.setText(text)

def beep():
     winsound.Beep(1000, 500)

def bad_posture(space_left, space_rigth):
    global threshold_start
    if (space_left or space_rigth) < space_threshold:
        if threshold_start is None:
            threshold_start = time.time()
        elif time.time() - threshold_start >= time_threshold:
            threading.Thread(target=beep, daemon=True).start()
            threshold_start = None
    else:
        threshold_start = None

def start_pyqt():
    global app, window
    app = QApplication(sys.argv)
    window = CoordinatesWindow()
    window.show()
    app.exec_()

thread_pyqt = threading.Thread(target=start_pyqt, daemon=True)
thread_pyqt.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = posture.process(rgb_frame)

    # Points
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape
        # Left Eye Point
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
        x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
        cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)
        # Right Eye Point
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
        x2, y2 = int(right_eye.x * w), int(right_eye.y * h)
        cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)
        # Left Shoulder
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        x3, y3 = int(left_shoulder.x * w), int(left_shoulder.y * h)
        cv2.circle(frame, (x3, y3), 5, (0, 255, 255), -1)
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        x4, y4 = int(right_shoulder.x * w), int(right_shoulder.y * h)
        cv2.circle(frame, (x4, y4), 5, (0, 255, 255), -1)

        # Show Coordinates
        window.update_text(f"""
        Left Eye:                  {x1}   {y1}
        Rigth Eye:                {x2}    {y2}
        Left Shoulder:          {x3}    {y3}
        Right Shoulder:        {x4}    {y4}
        Space Between Right:    {y4-y2}
        Space Bewteen Left:        {y3-y1}

                        """)
        bad_posture(y4-y2, y3-y1)


    cv2.imshow("Posture Check", frame)
    
    # Quit with 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

