img_sudoku = cv2.imread('images/sudoku.png', cv2.IMREAD_GRAYSCALE)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes[0, 0].imshow(img_sudoku, cmap='gray'); axes[0, 0].set_title('Original'); axes[0, 0].axis('off')

res1 = cv2.adaptiveThreshold(img_sudoku, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
axes[0, 1].imshow(res1, cmap='gray'); axes[0, 1].set_title('Mean, block=5, C=2'); axes[0, 1].axis('off')

res2 = cv2.adaptiveThreshold(img_sudoku, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)
axes[0, 2].imshow(res2, cmap='gray'); axes[0, 2].set_title('Mean, block=21, C=2'); axes[0, 2].axis('off')

res3 = cv2.adaptiveThreshold(img_sudoku, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
axes[0, 3].imshow(res3, cmap='gray'); axes[0, 3].set_title('Mean, block=11, C=10'); axes[0, 3].axis('off')

res4 = cv2.adaptiveThreshold(img_sudoku, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
axes[1, 0].imshow(res4, cmap='gray'); axes[1, 0].set_title('Gaussian, block=5, C=2'); axes[1, 0].axis('off')

res5 = cv2.adaptiveThreshold(img_sudoku, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
axes[1, 1].imshow(res5, cmap='gray'); axes[1, 1].set_title('Gaussian, block=21, C=2'); axes[1, 1].axis('off')

res6 = cv2.adaptiveThreshold(img_sudoku, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)
axes[1, 2].imshow(res6, cmap='gray'); axes[1, 2].set_title('Gaussian, block=11, C=10'); axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

cv2.imwrite('outputs/task2_exp1.png', res1)
cv2.imwrite('outputs/task2_exp2.png', res2)
cv2.imwrite('outputs/task2_exp3.png', res3)
cv2.imwrite('outputs/task2_exp4.png', res4)
cv2.imwrite('outputs/task2_exp5.png', res5)
cv2.imwrite('outputs/task2_exp6.png', res6)

print("1. Neighbourhood too small: Creates noise and broken characters because local statistics are too sensitive to individual pixels.")
print("2. Neighbourhood too large: Fails to adapt to local illumination changes, effectively acting like a global threshold.")
print("3. Increasing C: Subtracts a larger constant from the local mean. This reduces noise but can cause the text to become thinner or disappear.")
print("4. Cleanest combination: blockSize=11, C=2 (Mean or Gaussian) often provides a good balance.")
print("5. Better method: Gaussian-based is usually better as it weights nearby pixels more heavily, reducing noise compared to Mean.")
