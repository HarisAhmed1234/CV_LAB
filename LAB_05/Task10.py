img_target = cv2.imread('images/smarties.png')
gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

_, otsu_res = cv2.threshold(gray_target, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

hsv_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2HSV)
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])
hsv_res = cv2.inRange(hsv_target, lower_green, upper_green)

pixel_values = img_target.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixel_values, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
kmeans_res = centers[labels.flatten()].reshape(img_target.shape)

show_images(['Original', 'Method 1: Otsu', 'Method 2: HSV Green', 'Method 3: K-Means (K=3)'], 
            [img_target, otsu_res, hsv_res, kmeans_res], figsize=(20, 5))

cv2.imwrite('outputs/task10_otsu.png', otsu_res)
cv2.imwrite('outputs/task10_hsv.png', hsv_res)
cv2.imwrite('outputs/task10_kmeans.png', kmeans_res)

print("Method 1 (Otsu): Assumption: Bimodal histogram (objects vs background). Success: Captures all candies. Failure: Loses color information, merges shadows. Key Parameter: Global threshold value.")
print("Method 2 (HSV): Assumption: Distinct color range. Success: Isolates specific green candies perfectly. Failure: Misses other colored candies. Key Parameter: Lower/Upper HSV bounds.")
print("Method 3 (K-Means): Assumption: Color clusters. Success: Segments different colored candies into distinct regions. Failure: May merge similar colors (e.g., red and orange). Key Parameter: K (number of clusters).")
print("Conclusion: K-Means produced the most meaningful regions for this particular image because the image contains multiple distinct colors. Otsu struggled because the candies and background have overlapping intensity distributions, while HSV was too restrictive to one color.")
