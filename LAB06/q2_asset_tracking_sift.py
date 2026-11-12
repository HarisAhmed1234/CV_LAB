import cv2
import numpy as np

# Asset tracking in a computer lab using SIFT
scene = cv2.imread("lab.jpg")
scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

# Reference images of components to track
templates = {
    "Monitor": "monitor.jpg",
    "Keyboard": "keyboard.jpg",
    "Mouse": "mouse.jpg",
}

sift = cv2.SIFT_create()
kp_s, des_s = sift.detectAndCompute(scene_gray, None)
bf = cv2.BFMatcher()
inventory = {}
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

for idx, (name, path) in enumerate(templates.items()):
    ref = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    kp_r, des_r = sift.detectAndCompute(ref, None)
    matches = bf.knnMatch(des_r, des_s, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    found = False
    if len(good) >= 10:
        src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None and mask.sum() >= 10:
            h, w = ref.shape
            box = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(box, H)
            c = colors[idx % len(colors)]
            cv2.polylines(scene, [np.int32(proj)], True, c, 3)
            x, y = np.int32(proj[0][0])
            cv2.putText(scene, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
            found = True
    inventory[name] = "Found" if found else "Not found"

print("Inventory:")
for k, v in inventory.items():
    print(f"  {k}: {v}")

cv2.imshow("Asset Tracking", scene)
cv2.waitKey(0)
cv2.destroyAllWindows()
