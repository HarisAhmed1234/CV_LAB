import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a rectangular book cover
book = np.full((300, 220, 3), 255, dtype=np.uint8)

cv2.rectangle(book, (5, 5), (214, 294),
              (0, 0, 0), 3)

cv2.putText(book, "COMPUTER", (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 0, 0), 2)

cv2.putText(book, "VISION", (45, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0, 0, 255), 2)

# Create the background canvas
canvas = np.full((400, 400, 3), 180, dtype=np.uint8)

# Original corners of the book
src = np.float32([
    [0, 0],
    [219, 0],
    [219, 299],
    [0, 299]
])

# New positions of the corners
dst = np.float32([
    [100, 40],
    [330, 90],
    [290, 350],
    [60, 300]
])

# Apply perspective distortion
H1 = cv2.getPerspectiveTransform(src, dst)

tilted = cv2.warpPerspective(
    book, H1, (400, 400),
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(180, 180, 180)
)

# Define the corrected rectangular corners
src_correct = dst.copy()

dst_correct = np.float32([
    [0, 0],
    [219, 0],
    [219, 299],
    [0, 299]
])

# Calculate the correction matrix
H2 = cv2.getPerspectiveTransform(
    src_correct, dst_correct
)

# Restore the original shape
corrected = cv2.warpPerspective(
    tilted, H2, (220, 300)
)

# Display all three stages
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(book, cv2.COLOR_BGR2RGB))
plt.title("Original Book")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(tilted, cv2.COLOR_BGR2RGB))
plt.title("Tilted Book")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
plt.title("Corrected Book")
plt.axis("off")

plt.tight_layout()
plt.show()
