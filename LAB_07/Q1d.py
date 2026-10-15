import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a white image
img = np.full((400, 400, 3), 255, dtype=np.uint8)

# Draw a rectangle and a diagonal line
cv2.rectangle(img, (80, 100), (250, 280),
              (255, 0, 0), 3)

cv2.line(img, (80, 100), (250, 280),
         (0, 0, 255), 3)

# Three original points
src_pts = np.float32([
    [80, 100],
    [250, 100],
    [80, 280]
])

# Their corresponding destination points
dst_pts = np.float32([
    [100, 50],
    [280, 100],
    [50, 300]
])

# Calculate the affine matrix
M = cv2.getAffineTransform(src_pts, dst_pts)

print("Affine transformation matrix:\n", M)

# Apply the transformation
h, w = img.shape[:2]
result = cv2.warpAffine(img, M, (w, h))

# Display the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("After Affine Transformation")
plt.axis("off")

plt.tight_layout()
plt.show()
