import cv2
import numpy as np

# Load camera calibration parameters
data = np.load("camera_calibration.npz")

camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# Start the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Read and process frames
while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # Remove lens distortion
    undistorted = cv2.undistort(
        frame,
        camera_matrix,
        dist_coeffs
    )

    # Display both video feeds
    cv2.imshow("Original Feed", frame)
    cv2.imshow("Undistorted Feed", undistorted)

    # Press q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
