import numpy as np

# Original point
point = np.array([3, 4])

# Add the homogeneous coordinate
point_h = np.array([3, 4, 1])

print("Original point:", point)
print("Homogeneous coordinates:", point_h)

# Translation matrix: 5 units in X and 2 in Y
T = np.array([
    [1, 0, 5],
    [0, 1, 2],
    [0, 0, 1]
])

# Transform the point
result_h = T @ point_h

# Convert back to 2D coordinates
result = result_h[:2] / result_h[2]

print("Transformed homogeneous point:", result_h)
print("Final 2D point:", result)
