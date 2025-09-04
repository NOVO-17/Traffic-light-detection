import cv2
import numpy as np

def detect_color(mask, color_name, frame, box_color):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:  # filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, color_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
            detected = True
    return detected

__path__=r"e:\dev\ML\Traffic light detection\resources\1.mp4"
cap = cv2.VideoCapture(__path__)# 0 = webcam, __path__ -> video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #print(hsv)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    lower_green = np.array([40, 100, 100])
    upper_green = np.array([90, 255, 255])

    #masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    #clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    if detect_color(mask_red, "RED", frame, (0, 0, 255)):
        print("STOP (RED LIGHT)")
    elif detect_color(mask_yellow, "YELLOW", frame, (0, 255, 255)):
        print("READY (YELLOW LIGHT)")
    elif detect_color(mask_green, "GREEN", frame, (0, 255, 0)):
        print("GO (GREEN LIGHT)")

    cv2.imshow("Traffic Light Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27: # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

