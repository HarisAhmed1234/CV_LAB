import cv2
import numpy as np

def apply_perspective_transform(img, src_pts):

    # Get image dimensions
    h, w = img.shape[:2]

    # Define the output rectangle
    dst_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])

    # Calculate the transformation matrix
    M = cv2.getPerspectiveTransform(
        np.float32(src_pts),
        dst_pts
    )

    # Transform the image
    corrected = cv2.warpPerspective(
        img, M, (w, h)
    )

    return corrected
