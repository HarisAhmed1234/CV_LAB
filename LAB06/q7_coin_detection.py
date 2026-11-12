import cv2
import numpy as np

# Coin detection and counting using Hough Circle Transform
img = cv2.imread("coins.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                           param1=100, param2=40, minRadius=15, maxRadius=100)

out = img.copy()
count = 0
if circles is not None:
    circles = np.uint16(np.around(circles[0]))
    count = len(circles)
    for i, (x, y, r) in enumerate(circles, 1):
        cv2.circle(out, (x, y), r, (0, 255, 0), 3)
        cv2.circle(out, (x, y), 2, (0, 0, 255), 3)
        cv2.putText(out, str(i), (x - 8, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.putText(out, f"Coins: {count}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
print("Total coins detected:", count)

cv2.imshow("Coins", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
