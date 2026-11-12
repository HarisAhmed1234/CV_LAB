import cv2
import matplotlib.pyplot as plt

image = cv2.imread("Sukuna.jpeg")
if image is None:
    raise SystemExit("Error: Image not found.")

h, w = image.shape[:2]
band_top = int(h * 0.80)   # bottom 20%

# Solid blue rectangle on a copy of the image
overlay = image.copy()
cv2.rectangle(overlay, (0, band_top), (w, h), (255, 0, 0), -1)   # BGR blue

# Semi-transparent blend
alpha = 0.6
blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# Title over the transparent box
text = "King of Curses"
font = cv2.FONT_HERSHEY_SIMPLEX
scale = w / 600
thickness = max(2, int(scale * 2))
(tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
x = (w - tw) // 2
y = band_top + (h - band_top + th) // 2
cv2.putText(blended, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original"); plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
plt.title("Alpha-blended caption"); plt.axis("off")
plt.tight_layout()
plt.show()
