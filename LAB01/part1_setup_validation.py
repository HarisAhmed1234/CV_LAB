# In a Jupyter notebook, run these magic commands in the first cell:
#   %pip install opencv-python
#   %pip install matplotlib

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Validation: versions, then a tiny read/plot test
print("OpenCV version    :", cv2.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("NumPy version     :", np.__version__)

test = np.zeros((100, 100, 3), dtype=np.uint8)
test[:, :50] = (0, 0, 255)   # BGR red
test[:, 50:] = (255, 0, 0)   # BGR blue
plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
plt.title("Validation: left should be red, right should be blue")
plt.axis("off")
plt.show()
print("Environment OK")
