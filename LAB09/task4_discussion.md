# Task 4: Which model is best for what?

Fill the table from `output/model_comparison.csv` (produced by `task2_3_model_comparison.py`) on your own machine.

| Model | FPS | Inference (ms) | Mean confidence | Size (MB) |
|---|---|---|---|---|
| YOLOv3 | | | | |
| SSD (MobileNetV3) | | | | |
| RetinaNet (ResNet50-FPN) | | | | |

## Real-time detection
SSD with a MobileNet backbone (or a small YOLO) is the best fit. It predicts boxes and classes in a single forward pass from lightweight feature maps, so it gives the highest FPS. YOLOv3 is next, as a single-pass detector with a heavier Darknet-53 backbone. RetinaNet is also one-stage but its ResNet50 + FPN backbone is the slowest of the three.

## Small object recognition
RetinaNet is the strongest. The Feature Pyramid Network provides high-resolution feature maps with strong semantics, and Focal Loss reduces the dominance of easy background examples. YOLOv3 predicts at three scales and also handles small objects reasonably. Plain SSD, especially the low-resolution 320×320 MobileNet version, is weakest on small objects because its early feature maps are weak and the input is small.

## Resource-limited systems (CPU only)
SSD-MobileNet wins, since it has the fewest parameters, the smallest model size and the lowest latency. YOLOv3 and RetinaNet are much larger and need a GPU for real-time speed.

## Summary
- Speed and low resource use: SSD (MobileNet).
- Balanced accuracy and speed: YOLOv3 (or a newer YOLO).
- Best accuracy, especially on small objects, at the cost of speed: RetinaNet.

Back these claims with your measured FPS, confidence and size numbers in the table above, and note any case where your results differ from the expectation.
