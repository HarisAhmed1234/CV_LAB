import cv2
import numpy as np

# Object recognition in video using SIFT
ref = cv2.imread("reference.jpg", cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
kp_r, des_r = sift.detectAndCompute(ref, None)
h, w = ref.shape

flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
cap = cv2.VideoCapture("video.mp4")  # use 0 for webcam

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_f, des_f = sift.detectAndCompute(gray, None)

    if des_f is not None and len(kp_f) > 2:
        matches = flann.knnMatch(des_r, des_f, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2)
                if m.distance < 0.7 * n.distance]

        if len(good) >= 10:
            src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is not None:
                box = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                proj = cv2.perspectiveTransform(box, H)
                cv2.polylines(frame, [np.int32(proj)], True, (0, 255, 0), 3)
                cv2.putText(frame, "Object", tuple(np.int32(proj[0][0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Object Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
