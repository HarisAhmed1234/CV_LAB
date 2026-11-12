import cv2
import matplotlib.pyplot as plt

image_path = "Sukuna.jpeg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Side-by-side: wrong colours (BGR shown as-is) vs correct RGB
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("BGR shown directly (wrong)", fontsize=12, color="gray", pad=10)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_rgb)
    plt.title("King of Curses - Ryomen Sukuna", fontsize=16, color="darkred", pad=15)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
