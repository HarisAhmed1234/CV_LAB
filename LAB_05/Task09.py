img_dog = cv2.imread('images/dog.png')

def kmeans_segmentation(image, K):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    return segmented_image.reshape(image.shape)

seg_k2 = kmeans_segmentation(img_dog, 2)
seg_k4 = kmeans_segmentation(img_dog, 4)
seg_k6 = kmeans_segmentation(img_dog, 6)

show_images(['Original', 'K=2', 'K=4', 'K=6'], 
            [img_dog, seg_k2, seg_k4, seg_k6], figsize=(20, 5))

cv2.imwrite('outputs/task9_k2.png', seg_k2)
cv2.imwrite('outputs/task9_k4.png', seg_k4)
cv2.imwrite('outputs/task9_k6.png', seg_k6)

print("K=2: Major regions are separated into foreground and background. Heavy simplification.")
print("K=4: Captures more details (e.g., separates the dog from the sky, sand, and ocean). Moderate simplification.")
print("K=6: Further splits regions (e.g., splitting the sky or dog's fur). Least simplification, closer to original but still blocky.")
