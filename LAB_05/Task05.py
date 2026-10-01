img_smarties = cv2.imread('images/smarties.png', cv2.IMREAD_GRAYSCALE)

edges1 = cv2.Canny(img_smarties, 50, 150)
edges2 = cv2.Canny(img_smarties, 100, 200)
edges3 = cv2.Canny(img_smarties, 150, 250)

show_images(['Original', 'Low=50, High=150', 'Low=100, High=200', 'Low=150, High=250'], 
            [img_smarties, edges1, edges2, edges3], figsize=(20, 5))

cv2.imwrite('outputs/task5_edges1.png', edges1)
cv2.imwrite('outputs/task5_edges2.png', edges2)
cv2.imwrite('outputs/task5_edges3.png', edges3)

print("Strong edges: Found in all three results (e.g., candy boundaries against the background).")
print("Weak edges: Found in low-threshold results but disappear as the high threshold increases.")
print("Missing edges: High thresholds may miss subtle boundaries or low-contrast edges.")
print("Unwanted edges: Low thresholds pick up noise and texture (e.g., reflections on the candies).")
print("Effect of thresholds: Increasing the high threshold makes the detector more selective, keeping only the strongest edges. The low threshold controls the hysteresis linking; if it's too high, weak edge segments are lost.")
