img_coins = cv2.imread('images/water_coins.jpg')
gray_coins = cv2.cvtColor(img_coins, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray_coins, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(img_coins, markers)
img_coins[markers == -1] = [0, 0, 255]

show_images(['Original', 'Threshold', 'Sure Background', 'Distance Transform', 'Sure Foreground', 'Unknown', 'Watershed Result'], 
            [cv2.cvtColor(cv2.imread('images/water_coins.jpg'), cv2.COLOR_BGR2RGB), thresh, sure_bg, dist_transform, sure_fg, unknown, cv2.cvtColor(img_coins, cv2.COLOR_BGR2RGB)], 
            figsize=(24, 4))

cv2.imwrite('outputs/task7_watershed.png', img_coins)
