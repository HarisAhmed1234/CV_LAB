import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE = 800
canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
center = (SIZE // 2, SIZE // 2)   # (x, y)

# Target board: 5 concentric circles drawn from largest to smallest (BGR colours)
radii = [350, 280, 210, 140, 70]
colors = [(0, 0, 255), (255, 255, 255), (0, 0, 255), (255, 255, 255), (0, 0, 255)]
for r, c in zip(radii, colors):
    cv2.circle(canvas, center, r, c, -1)

# Bounding box around the outermost circle
r0 = radii[0]
top_left = (center[0] - r0, center[1] - r0)
bottom_right = (center[0] + r0, center[1] + r0)
cv2.rectangle(canvas, top_left, bottom_right, (0, 255, 0), 3)

print("Canvas size:", canvas.shape[:2])
print("Canvas center (x, y):", center)
print("Bounding box:", top_left, "->", bottom_right)

plt.figure(figsize=(7, 7))
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.title(f"Target board, center = {center}")
plt.axis("off")
plt.show()
