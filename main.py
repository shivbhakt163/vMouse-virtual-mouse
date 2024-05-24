import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller
import numpy as np

mouse = Controller()
cp = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = 1920, 1080
num_positions = 10
positions_x = []
positions_y = []

def calculate_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

while True:
    ret, frame = cp.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(rgb_frame)
    hands = result.multi_hand_landmarks
    
    if hands:
        x4, y4, x8, y8, x12, y12 = None, None, None, None, None, None
        
        for hand_landmarks in hands:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * screen_width)
                y = int(landmark.y * screen_height)
                if id == 8:
                    x8, y8 = x, y
                    positions_x.append(x)
                    positions_y.append(y)
                    if len(positions_x) > num_positions:
                        positions_x.pop(0)
                        positions_y.pop(0)
                    avg_x = int(np.mean(positions_x))
                    avg_y = int(np.mean(positions_y))
                    mouse.position = (avg_x, avg_y)
                elif id == 12:
                    x12, y12 = x, y
                elif id == 4:
                    x4, y4 = x, y
        if x8 is not None and y8 is not None and x12 is not None and y12 is not None:
            distance = calculate_distance(x8, y8, x12, y12)
            if distance <= 175:
                mouse.click(Button.left)
        if x4 is not None and y4 is not None and x8 is not None and y8 is not None:
            distance1 = calculate_distance(x8, y8, x4, y4)
            if distance1 <= 175:
                mouse.press(Button.left)
            else:
                mouse.release(Button.left)
        if x12 is not None and y12 is not None and x4 is not None and y4 is not None:
            distance2 = calculate_distance(x12, y12, x4, y4)
            if distance2 <= 175:
                mouse.click(Button.right)
    cv2.imshow("Processing Data... ", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('g'):
        break
cp.release()
cv2.destroyAllWindows()
