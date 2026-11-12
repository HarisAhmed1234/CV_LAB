import cv2
import numpy as np

# Smart security system: boundary detection inside a security zone
cap = cv2.VideoCapture(0)  # or a video file path
ok, frame = cap.read()
h, w = frame.shape[:2]

# Predefined security zone (x1, y1, x2, y2)
zone = (int(0.3 * w), int(0.3 * h), int(0.7 * w), int(0.8 * h))
zx1, zy1, zx2, zy2 = zone

bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
kernel = np.ones((5, 5), np.uint8)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    mask = bg.apply(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Only look at the security zone
    zone_mask = np.zeros_like(mask)
    zone_mask[zy1:zy2, zx1:zx2] = mask[zy1:zy2, zx1:zx2]

    contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    alarm = False
    for c in contours:
        if cv2.contourArea(c) > 1500:
            alarm = True
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 2)
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 255), 2)

    color = (0, 0, 255) if alarm else (0, 255, 0)
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color, 2)
    if alarm:
        cv2.putText(frame, "ALARM: UNAUTHORIZED OBJECT", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        print("\a ALARM triggered")

    cv2.imshow("Security", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
