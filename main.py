import cv2
import numpy as np

def detect_color(mask, color_name, frame, box_color, stats_counter):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, color_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
            detected = True

    if detected:
        stats_counter[color_name] += 1
    return detected

__path__=r"e:\dev\ML\Traffic light detection\resources\1.mp4"
cap = cv2.VideoCapture(__path__)# 0 = webcam, __path__ -> video

stats = {
    "total_frames": 0,
    "detected_frames": 0,
    "RED": 0,
    "YELLOW": 0,
    "GREEN": 0
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    stats["total_frames"] += 1

    frame = cv2.resize(frame, (640, 480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    lower_yellow = np.array([20, 100, 100])   
    upper_yellow = np.array([35, 255, 255])

    lower_green = np.array([40, 100, 100])
    upper_green = np.array([90, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    detected_any = False
    if detect_color(mask_red, "RED", frame, (0, 0, 255), stats):
        print("STOP (RED LIGHT)")
        detected_any = True
    elif detect_color(mask_yellow, "YELLOW", frame, (0, 255, 255), stats):
        print("READY (YELLOW LIGHT)")
        detected_any = True
    elif detect_color(mask_green, "GREEN", frame, (0, 255, 0), stats):
        print("GO (GREEN LIGHT)")
        detected_any = True

    if detected_any:
        stats["detected_frames"] += 1

    cv2.imshow("Traffic Light Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

# Accuracy Report
print("\n========== Detection Accuracy Report ==========")
print(f"Total Frames Processed: {stats['total_frames']}")
print(f"Frames with Detection:  {stats['detected_frames']}")
if stats["total_frames"] > 0:
    accuracy = (stats["detected_frames"] / stats["total_frames"]) * 100
    print(f"Overall Detection Accuracy: {accuracy:.2f}%")
print(f"Red Detections:    {stats['RED']}")
print(f"Yellow Detections: {stats['YELLOW']}")
print(f"Green Detections:  {stats['GREEN']}")
print("==============================================")