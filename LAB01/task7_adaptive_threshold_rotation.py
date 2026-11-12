import cv2
import matplotlib.pyplot as plt

# A grayscale document or high-contrast image
gray = cv2.imread("document.jpg", cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise SystemExit("Error: Image not found.")

_, global_th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
adaptive_th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

# Rotate the adaptive output by 45 degrees, scale 0.8 so corners are not cut off
h, w = adaptive_th.shape
M = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 0.8)
rotated = cv2.warpAffine(adaptive_th, M, (w, h))

plt.figure(figsize=(16, 5))
for i, (im, title) in enumerate([(gray, "Original grayscale"),
                                 (global_th, "Global binary (127)"),
                                 (adaptive_th, "Adaptive Gaussian"),
                                 (rotated, "Adaptive rotated 45 deg, scale 0.8")], 1):
    plt.subplot(1, 4, i)
    plt.imshow(im, cmap="gray")
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
plt.show()
