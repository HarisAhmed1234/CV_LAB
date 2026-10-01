img_mri = cv2.imread('images/brain_mri.png', cv2.IMREAD_GRAYSCALE)

def region_growing(image, seed, threshold):
    mask = np.zeros_like(image, dtype=np.uint8)
    stack = [seed]
    seed_intensity = image[seed[0], seed[1]]
    while stack:
        x, y = stack.pop()
        if x < 0 or x >= image.shape[0] or y < 0 or y >= image.shape[1]:
            continue
        if mask[x, y] == 0 and abs(int(image[x, y]) - int(seed_intensity)) <= threshold:
            mask[x, y] = 255
            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
    return mask

seed1 = (150, 100) 
seed2 = (100, 150) 

mask1 = region_growing(img_mri, seed1, 20)
mask2 = region_growing(img_mri, seed1, 40)
mask3 = region_growing(img_mri, seed1, 60)
mask4 = region_growing(img_mri, seed2, 40)

show_images(['Original', 'Seed1 T=20', 'Seed1 T=40', 'Seed1 T=60', 'Seed2 T=40'], 
            [img_mri, mask1, mask2, mask3, mask4], figsize=(20, 5))

cv2.imwrite('outputs/task6_mask1.png', mask1)
cv2.imwrite('outputs/task6_mask2.png', mask2)
cv2.imwrite('outputs/task6_mask3.png', mask3)
cv2.imwrite('outputs/task6_mask4.png', mask4)

print("Why changing the seed point changes the region: Region growing is a local algorithm. It only expands from the seed based on similarity. If the seed is in the tumor, it grows the tumor. If the seed is in healthy tissue, it grows healthy tissue. If the seed is on a boundary, it might leak into both.")
