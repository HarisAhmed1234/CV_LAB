img_sudoku = cv2.imread('images/sudoku.png', cv2.IMREAD_GRAYSCALE)

_, g1 = cv2.threshold(img_sudoku, 100, 255, cv2.THRESH_BINARY)
_, g2 = cv2.threshold(img_sudoku, 150, 255, cv2.THRESH_BINARY)
_, g3 = cv2.threshold(img_sudoku, 200, 255, cv2.THRESH_BINARY)

adaptive = cv2.adaptiveThreshold(img_sudoku, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

show_images(['Original', 'Global T=100', 'Global T=150', 'Global T=200', 'Adaptive'], 
            [img_sudoku, g1, g2, g3, adaptive], figsize=(20, 4))

cv2.imwrite('outputs/task1_global_100.png', g1)
cv2.imwrite('outputs/task1_global_150.png', g2)
cv2.imwrite('outputs/task1_global_200.png', g3)
cv2.imwrite('outputs/task1_adaptive.png', adaptive)

print("Why a single threshold struggles: Uneven lighting causes the pixel intensity of the text to vary across the image. A threshold that is too high (200) will miss text in dark areas. A threshold that is too low (100) will pick up unwanted background noise in bright areas. Adaptive thresholding solves this by calculating a local threshold for each pixel based on its neighborhood.")
