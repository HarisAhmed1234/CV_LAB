img_coins = cv2.imread('images/water_coins.jpg', cv2.IMREAD_GRAYSCALE)

otsu_thresh, otsu_mask = cv2.threshold(img_coins, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
hist = cv2.calcHist([img_coins], [0], None, [256], [0, 256])

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img_coins, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.plot(hist, color='black')
plt.title(f'Histogram (Otsu T={otsu_thresh:.2f})')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.imshow(otsu_mask, cmap='gray')
plt.title('Otsu Binary Mask')
plt.axis('off')
plt.tight_layout()
plt.show()

cv2.imwrite('outputs/task3_otsu_mask.png', otsu_mask)

alpha = 1.5
adjusted = cv2.convertScaleAbs(img_coins, alpha=alpha, beta=0)
otsu_thresh_adj, otsu_mask_adj = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Original Otsu Threshold: {otsu_thresh:.2f}")
print(f"Adjusted Contrast Otsu Threshold: {otsu_thresh_adj:.2f}")
print("Why Otsu is useful: Otsu automatically finds the optimal threshold by maximizing the variance between the two classes (foreground and background). It removes the need for manual trial-and-error when the programmer does not know the correct threshold beforehand.")
