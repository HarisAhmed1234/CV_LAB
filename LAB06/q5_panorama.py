import cv2
import numpy as np

# Panorama stitching with SIFT (images listed left to right)
paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
images = [cv2.imread(p) for p in paths]

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()


def stitch_pair(base, new):
    """Warp `new` onto `base` using SIFT matches and a RANSAC homography."""
    g1 = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)

    matches = bf.knnMatch(d2, d1, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        raise RuntimeError("Not enough matches to stitch")

    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    h1, w1 = base.shape[:2]
    h2, w2 = new.shape[:2]
    corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    all_pts = np.concatenate([np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2),
                              warped_corners])
    xmin, ymin = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(all_pts.max(axis=0).ravel() + 0.5)

    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    result = cv2.warpPerspective(new, T @ H, (xmax - xmin, ymax - ymin))
    result[-ymin:h1 - ymin, -xmin:w1 - xmin] = base
    return result


pano = images[0]
for img in images[1:]:
    pano = stitch_pair(pano, img)

cv2.imwrite("panorama.jpg", pano)
cv2.imshow("Panorama", pano)
cv2.waitKey(0)
cv2.destroyAllWindows()
