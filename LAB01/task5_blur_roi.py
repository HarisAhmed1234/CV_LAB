import cv2
import matplotlib.pyplot as plt

image = cv2.imread("Sukuna.jpeg")
if image is None:
    raise SystemExit("Error: Image not found.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
blurred = cv2.GaussianBlur(image_rgb, (25, 25), 0)

h, w = image_rgb.shape[:2]
ROI = 300
if h < ROI or w < ROI:
    raise SystemExit(f"Image is {w}x{h}; need at least {ROI}x{ROI}.")

y0, x0 = (h - ROI) // 2, (w - ROI) // 2
roi_original = image_rgb[y0:y0 + ROI, x0:x0 + ROI]
roi_blurred = blurred[y0:y0 + ROI, x0:x0 + ROI]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(roi_original)
plt.title("Original ROI (300x300)", fontsize=14, color="navy")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(roi_blurred)
plt.title("Gaussian Blur 25x25 ROI", fontsize=14, color="darkred")
plt.axis("off")

plt.tight_layout()
plt.show()
