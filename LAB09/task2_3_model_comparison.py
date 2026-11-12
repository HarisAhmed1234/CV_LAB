"""
Tasks 2 and 3: run YOLOv3, SSD and RetinaNet on the same input and measure
FPS, inference time per frame, confidence scores and model size.

pip install ultralytics torch torchvision opencv-python pandas
SOURCE can be an image ("test.jpg"), a video ("video.mp4") or 0 for the webcam.
"""
import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.models import detection as det
from ultralytics import YOLO

SOURCE = "test.jpg"
N_FRAMES = 30            # frames to time (an image is repeated N_FRAMES times)
THRESH = 0.5
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ---- Frame source ----
def get_frames(source, n):
    if isinstance(source, str) and source.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(source)
        return [img.copy() for _ in range(n)]
    cap = cv2.VideoCapture(source)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


# ---- Detectors: each returns a list of (x1, y1, x2, y2, label, score) ----
class YoloV3:
    name = "YOLOv3"

    def __init__(self):
        self.model = YOLO("yolov3u.pt")          # pretrained YOLOv3 (Ultralytics build)
        self.model_params = sum(p.numel() for p in self.model.model.parameters())

    def __call__(self, bgr):
        r = self.model(bgr, conf=THRESH, verbose=False)[0]
        return [(*map(int, b.xyxy[0]), self.model.names[int(b.cls[0])], float(b.conf[0]))
                for b in r.boxes]


class TorchDetector:
    def __init__(self, name, builder, weights):
        self.name = name
        self.weights = weights
        self.model = builder(weights=weights).eval().to(device)
        self.categories = weights.meta["categories"]
        self.model_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)
        out = self.model([t])[0]
        res = []
        for b, l, s in zip(out["boxes"], out["labels"], out["scores"]):
            if s >= THRESH:
                x1, y1, x2, y2 = map(int, b.tolist())
                res.append((x1, y1, x2, y2, self.categories[int(l)], float(s)))
        return res


detectors = [
    YoloV3(),
    TorchDetector("SSD (MobileNetV3)", det.ssdlite320_mobilenet_v3_large,
                  det.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT),
    TorchDetector("RetinaNet (ResNet50-FPN)", det.retinanet_resnet50_fpn,
                  det.RetinaNet_ResNet50_FPN_Weights.DEFAULT),
]

frames = get_frames(0 if SOURCE == "0" else SOURCE, N_FRAMES)
print("Frames:", len(frames))

rows = []
for d in detectors:
    d(frames[0])                                 # warm-up (excluded from timing)
    times, confs, counts = [], [], []
    last = None
    for f in frames:
        t0 = time.perf_counter()
        dets = d(f)
        times.append(time.perf_counter() - t0)
        counts.append(len(dets))
        confs += [s for *_, s in dets]
        last = (f, dets)

    # Annotated output for the report
    img, dets = last
    img = img.copy()
    for x1, y1, x2, y2, lab, s in dets:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{lab} {s:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    fname = d.name.split()[0].lower()
    cv2.imwrite(f"{OUT_DIR}/{fname}_result.jpg", img)

    rows.append({
        "Model": d.name,
        "Avg inference (ms)": round(1000 * np.mean(times), 1),
        "FPS": round(1 / np.mean(times), 2),
        "Avg detections/frame": round(float(np.mean(counts)), 1),
        "Mean confidence": round(float(np.mean(confs)), 3) if confs else 0.0,
        "Parameters (M)": round(d.model_params / 1e6, 1),
        "Model size (MB, fp32)": round(d.model_params * 4 / 1e6, 1),
    })

table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
print("Saved annotated images and", f"{OUT_DIR}/model_comparison.csv")
