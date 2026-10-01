img_smarties = cv2.imread('images/smarties.png')
hsv_smarties = cv2.cvtColor(img_smarties, cv2.COLOR_BGR2HSV)

lower_restrictive = np.array([0, 150, 150])
upper_restrictive = np.array([10, 255, 255])
mask_restrictive = cv2.inRange(hsv_smarties, lower_restrictive, upper_restrictive)

lower_final = np.array([0, 80, 80])
upper_final = np.array([15, 255, 255])
mask_final = cv2.inRange(hsv_smarties, lower_final, upper_final)

extracted = cv2.bitwise_and(img_smarties, img_smarties, mask=mask_final)

show_images(['Original', 'Restrictive Mask', 'Final Mask', 'Extracted Color'], 
            [img_smarties, mask_restrictive, mask_final, extracted], figsize=(20, 5))

cv2.imwrite('outputs/task4_restrictive.png', mask_restrictive)
cv2.imwrite('outputs/task4_final.png', mask_final)
cv2.imwrite('outputs/task4_extracted.png', extracted)

print("Why the restrictive mask fails: It only captures the most saturated and brightest pixels of the object. Due to lighting variations, shadows, and highlights on the spherical candies, many pixels of the same object fall outside this narrow range, causing the mask to be fragmented and incomplete.")
