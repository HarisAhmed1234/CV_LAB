import cv2
import numpy as np
import matplotlib.pyplot as plt

# Make a sample image
img = np.full((400, 400, 3), 255, dtype=np.uint8)

cv2.rectangle(img, (100, 100), (200, 200),
              (255, 0, 0), -1)
cv2.circle(img, (150, 150), 25,
           (0, 255, 0), -1)

h, w = img.shape[:2]

# Translation: shift right by 50 and down by 30
M1 = np.float32([[1, 0, 50],
                 [0, 1, 30]])
translated = cv2.warpAffine(img, M1, (w, h))

# Rotation: 45 degrees around the center
center = (w // 2, h // 2)
M2 = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, M2, (w, h))

# Scaling: increase size by 1.5 times
scaled = cv2.resize(img, None, fx=1.5, fy=1.5,
                    interpolation=cv2.INTER_LINEAR)

# Horizontal shearing
M3 = np.float32([[1, 0.5, 0],
                 [0, 1, 0]])
sheared = cv2.warpAffine(img, M3, (w, h))

# Show the transformations
results = [
    ("Original Image", img),
    ("Translation", translated),
    ("Rotation", rotated),
    ("Scaling", scaled),
    ("Shearing", sheared)
]

plt.figure(figsize=(14, 8))

for i, (name, image) in enumerate(results):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(name)
    plt.axis("off")

plt.tight_layout()
plt.show()
