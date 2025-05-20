'''This Project is created by Pranat Pagar ppagar602@gmail.com''' 
import cv2
import mediapipe as mp
import pyautogui
import time

# Init
cap = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh()
screen_w, screen_h = pyautogui.size()

# Cooldown between key presses
cooldown = 0.6
last_press_time = 0
last_direction = None

# Grid config (in % of frame size)
box_margin = 0.05  # 15% margin to define center zone

def draw_grid(frame, frame_w, frame_h):
    # Draw vertical and horizontal lines
    third_w = frame_w // 3
    third_h = frame_h // 3
    for i in range(1, 3):
        cv2.line(frame, (i * third_w, 0), (i * third_w, frame_h), (200, 200, 200), 1)
        cv2.line(frame, (0, i * third_h), (frame_w, i * third_h), (200, 200, 200), 1)

    # Draw center box
    x1 = int(frame_w * (0.5 - box_margin))
    y1 = int(frame_h * (0.5 - box_margin))
    x2 = int(frame_w * (0.5 + box_margin))
    y2 = int(frame_h * (0.5 + box_margin))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return x1, y1, x2, y2

def detect_head_direction(nose_x, nose_y, frame_w, frame_h, x1, y1, x2, y2):
    direction = None
    if nose_x < x1:
        direction = "left"
    elif nose_x > x2:
        direction = "right"
    elif nose_y < y1:
        direction = "up"
    elif nose_y > y2:
        direction = "down"
    return direction

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    frame_h, frame_w, _ = frame.shape

    # Draw grid and center box
    x1, y1, x2, y2 = draw_grid(frame, frame_w, frame_h)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        nose_x = int(landmarks[1].x * frame_w)
        nose_y = int(landmarks[1].y * frame_h)

        # Draw nose point
        cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)

        # Determine direction
        direction = detect_head_direction(nose_x, nose_y, frame_w, frame_h, x1, y1, x2, y2)
        current_time = time.time()
        if direction and (direction != last_direction or current_time - last_press_time > cooldown):
            print(f"Head moved {direction}")
            pyautogui.press(direction)
            last_direction = direction
            last_press_time = current_time

        if not direction:
            last_direction = None  # Reset when head returns to center

    cv2.imshow("Head Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
