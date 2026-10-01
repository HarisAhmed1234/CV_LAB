img_coins = cv2.imread('images/water_coins.jpg')
gray_coins = cv2.cvtColor(img_coins, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_coins, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

thresholds = [0.1, 0.3, 0.5, 0.7]

print("Experiment | Distance Threshold | Foreground Markers | Separated Regions | Observed Result")
print("-" * 80)

for i, t in enumerate(thresholds):
    _, sure_fg = cv2.threshold(dist_transform, t * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    num_markers, _ = cv2.connectedComponents(sure_fg)
    num_markers = num_markers - 1 
    
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_coins.copy(), markers)
    
    unique_regions = len(np.unique(markers)) - 2 
    
    print(f"{i+1:9} | {t:18} | {num_markers:17} | {unique_regions:17} | {'Good' if num_markers == 10 else 'Merged/Split'}")

print("The parameter range of 0.3 to 0.5 usually gives the most meaningful separation. Too low (0.1) merges touching coins into a single marker. Too high (0.7) causes a single coin to be split into multiple markers.")
