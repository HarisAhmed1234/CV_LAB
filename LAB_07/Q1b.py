import numpy as np
import matplotlib.pyplot as plt

# Define the eight vertices of a cube
points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
], dtype=float)

# Define the twelve edges
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

# Rotation angles
ax = np.radians(30)
ay = np.radians(30)
az = np.radians(30)

# Rotation around X-axis
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(ax), -np.sin(ax)],
    [0, np.sin(ax), np.cos(ax)]
])

# Rotation around Y-axis
Ry = np.array([
    [np.cos(ay), 0, np.sin(ay)],
    [0, 1, 0],
    [-np.sin(ay), 0, np.cos(ay)]
])

# Rotation around Z-axis
Rz = np.array([
    [np.cos(az), -np.sin(az), 0],
    [np.sin(az), np.cos(az), 0],
    [0, 0, 1]
])

# Combine the rotations
R = Rz @ Ry @ Rx

# Rotate the cube vertices
rotated = points @ R.T

# Translate by (2, 1, 1)
translation = np.array([2, 1, 1])
transformed = rotated + translation

# Draw both cubes
fig = plt.figure(figsize=(12, 5))

for n, data, title in [
    (1, points, "Original Cube"),
    (2, transformed, "Transformed Cube")
]:
    ax3d = fig.add_subplot(1, 2, n, projection="3d")

    for i, j in edges:
        ax3d.plot(data[[i, j], 0],
                  data[[i, j], 1],
                  data[[i, j], 2])

    ax3d.set_title(title)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()
