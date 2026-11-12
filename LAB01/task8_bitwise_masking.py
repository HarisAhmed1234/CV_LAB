import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE = (500, 500)
img1 = cv2.imread("Sukuna.jpeg")
img2 = cv2.imread("Gojo.jpeg")
if img1 is None or img2 is None:
    raise SystemExit("Error: Image not found.")

img1 = cv2.resize(img1, SIZE)
img2 = cv2.resize(img2, SIZE)

# Binary mask: black image with a thick white circle in the centre
mask = np.zeros((SIZE[1], SIZE[0]), dtype=np.uint8)
cv2.circle(mask, (SIZE[0] // 2, SIZE[1] // 2), 180, 255, -1)

fg = cv2.bitwise_and(img1, img1, mask=mask)           # shape cut from image 1
mask_inv = cv2.bitwise_not(mask)                      # inverted mask
bg = cv2.bitwise_and(img2, img2, mask=mask_inv)       # background cut from image 2
combined = cv2.bitwise_or(fg, bg)

to_rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
items = [(to_rgb(img1), "Image 1"), (to_rgb(img2), "Image 2"), (mask, "Mask"),
         (mask_inv, "Inverted mask"), (to_rgb(fg), "AND: shape from image 1"),
         (to_rgb(bg), "AND: background from image 2"), (to_rgb(combined), "OR: combined")]

plt.figure(figsize=(16, 8))
for i, (im, title) in enumerate(items, 1):
    plt.subplot(2, 4, i)
    plt.imshow(im, cmap="gray" if im.ndim == 2 else None)
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
plt.show()
