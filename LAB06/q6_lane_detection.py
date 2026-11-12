import cv2
import numpy as np

# Lane detection using Hough Line Transform
img = cv2.imread("road.jpg")
h, w = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# Region of interest: trapezoid in front of the vehicle
mask = np.zeros_like(edges)
roi = np.array([[(0, h), (int(0.45 * w), int(0.6 * h)),
                 (int(0.55 * w), int(0.6 * h)), (w, h)]], np.int32)
cv2.fillPoly(mask, roi, 255)
masked = cv2.bitwise_and(edges, mask)

lines = cv2.HoughLinesP(masked, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=150)

left, right = [], []
if lines is not None:
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.4:
            continue
        fit = np.polyfit((x1, x2), (y1, y2), 1)
        (left if slope < 0 else right).append(fit)


def make_line(fits):
    if not fits:
        return None
    m, b = np.mean(fits, axis=0)
    y1, y2 = h, int(0.6 * h)
    return int((y1 - b) / m), y1, int((y2 - b) / m), y2


out = img.copy()
for fits in (left, right):
    l = make_line(fits)
    if l:
        cv2.line(out, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 8)

cv2.imshow("Lane Detection", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
