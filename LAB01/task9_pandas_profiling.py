import cv2
import pandas as pd

image = cv2.imread("Sukuna.jpeg")
if image is None:
    raise SystemExit("Error: Image not found.")

b, g, r = cv2.split(image)   # OpenCV order is B, G, R

# Flatten each 2D channel into a 1D array
df = pd.DataFrame({
    "Red": r.flatten(),
    "Green": g.flatten(),
    "Blue": b.flatten(),
})

print("Image shape:", image.shape)
print(df.describe())
