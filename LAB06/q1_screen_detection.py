import cv2
import numpy as np

# Screen detection in a computer lab using Hough Line Transform
img = cv2.imread("lab.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
edges = cv2.Canny(blur, 50, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                        minLineLength=60, maxLineGap=10)

h_lines, v_lines = [], []
if lines is not None:
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 10 or angle > 170:
            h_lines.append((x1, y1, x2, y2))
        elif 80 < angle < 100:
            v_lines.append((x1, y1, x2, y2))

out = img.copy()
for x1, y1, x2, y2 in h_lines:
    cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
for x1, y1, x2, y2 in v_lines:
    cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Screen boundary = rectangle formed by the extent of horizontal/vertical lines
# found inside each connected edge region
dil = cv2.dilate(edges, np.ones((15, 15), np.uint8))
contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
screens = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w > 80 and h > 60 and 1.2 < w / h < 2.2:
        screens.append((x, y, w, h))
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 3)

# Status: bright screen = on, dark screen = off
for i, (x, y, w, h) in enumerate(screens, 1):
    status = "ON" if gray[y:y + h, x:x + w].mean() > 90 else "OFF"
    cv2.putText(out, f"S{i}:{status}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

EXPECTED_SCREENS = 10  # set to the number of computers in your lab
if len(screens) < EXPECTED_SCREENS:
    print(f"Anomaly: {EXPECTED_SCREENS - len(screens)} screen(s) missing")
print("Screens detected:", len(screens))

cv2.imshow("Screen Detection", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
