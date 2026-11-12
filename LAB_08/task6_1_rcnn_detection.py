"""
Task 6.1: Object detection with R-CNN (classic pipeline) on a dataset of your choice (not BCCD).

Pipeline: selective search proposals -> warp each region to 224x224 -> CNN features
-> linear SVM per class (+ background) -> NMS -> draw boxes.

Dataset layout (same as the manual's train.csv):
    data/train_images/*.jpg   data/test_images/*.jpg   data/train.csv
    train.csv columns: image_names, cell_type (class), xmin, xmax, ymin, ymax
Requires: pip install opencv-contrib-python torch torchvision scikit-learn pandas numpy
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torchvision.ops import nms, box_iou
from sklearn.svm import LinearSVC

DATA = "data"
TRAIN_DIR, TEST_DIR = f"{DATA}/train_images", f"{DATA}/test_images"
OUT_DIR = "output"
MAX_PROPOSALS = 300          # proposals kept per image (the original uses ~2000)
POS_IOU, NEG_IOU = 0.5, 0.3  # IoU thresholds for positive / background regions
MAX_BG_PER_IMG = 30
SCORE_THRESH, NMS_IOU = 0.5, 0.3
os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 1. Annotations ----
df = pd.read_csv(f"{DATA}/train.csv")
classes = sorted(df["cell_type"].unique())
cls_id = {c: i + 1 for i, c in enumerate(classes)}   # 0 = background
print("Classes:", classes, "| train images:", df["image_names"].nunique())

# ---- 2. Region proposals (selective search) ----
def propose(img, k=MAX_PROPOSALS):
    h, w = img.shape[:2]
    scale = 500 / max(h, w)                      # speed: search on a smaller copy
    small = cv2.resize(img, None, fx=scale, fy=scale)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(small)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()[:k] / scale             # (x, y, w, h) in original coordinates
    boxes = np.array([[x, y, x + bw, y + bh] for x, y, bw, bh in rects], dtype=np.float32)
    return boxes

# ---- 3. CNN feature extractor (pretrained ResNet-18 without its classifier) ----
backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
backbone.fc = torch.nn.Identity()
backbone.eval().to(device)
prep = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@torch.no_grad()
def features(img_rgb, boxes, batch=64):
    feats = []
    for i in range(0, len(boxes), batch):
        crops = []
        for x1, y1, x2, y2 in boxes[i:i + batch].astype(int):
            crop = img_rgb[max(y1, 0):max(y2, y1 + 2), max(x1, 0):max(x2, x1 + 2)]
            crops.append(prep(crop))
        feats.append(backbone(torch.stack(crops).to(device)).cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, 512))

# ---- 4. Build the training set: positives (IoU>=0.5), background (IoU<0.3) ----
X, y = [], []
for name, g in df.groupby("image_names"):
    path = f"{TRAIN_DIR}/{name}"
    img = cv2.imread(path)
    if img is None:
        continue
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt = torch.tensor(g[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
    gt_cls = np.array([cls_id[c] for c in g["cell_type"]])

    props = propose(img)
    # ground-truth boxes are positives too
    props = np.vstack([props, gt.numpy()])
    iou = box_iou(torch.tensor(props), gt)
    best_iou, best_gt = iou.max(1)
    labels = np.where(best_iou.numpy() >= POS_IOU, gt_cls[best_gt.numpy()], -1)
    labels[best_iou.numpy() < NEG_IOU] = 0

    bg_idx = np.where(labels == 0)[0]
    bg_idx = np.random.choice(bg_idx, min(MAX_BG_PER_IMG, len(bg_idx)), replace=False)
    keep = np.concatenate([np.where(labels > 0)[0], bg_idx]).astype(int)
    if len(keep):
        X.append(features(rgb, props[keep]))
        y.append(labels[keep])

X, y = np.concatenate(X), np.concatenate(y)
print("Training regions:", X.shape, "| label counts:", np.bincount(y))

# ---- 5. Classify regions with a linear SVM (one-vs-rest incl. background) ----
svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
svm.fit(X, y)
print("SVM training accuracy: %.3f" % svm.score(X, y))

# ---- 6. Inference on test images ----
colors = {i + 1: c for i, c in enumerate([(0, 0, 255), (255, 0, 0), (0, 160, 0), (0, 200, 255), (255, 0, 255)])}
for fname in sorted(os.listdir(TEST_DIR)):
    img = cv2.imread(f"{TEST_DIR}/{fname}")
    if img is None:
        continue
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    props = propose(img)
    scores = svm.decision_function(features(rgb, props))   # (N, n_classes+1)
    out = img.copy()
    for c in range(1, len(classes) + 1):
        s = torch.tensor(scores[:, c])
        m = s > SCORE_THRESH
        if m.sum() == 0:
            continue
        b = torch.tensor(props)[m]
        for idx in nms(b, s[m], NMS_IOU):
            x1, y1, x2, y2 = b[idx].int().tolist()
            cv2.rectangle(out, (x1, y1), (x2, y2), colors.get(c, (0, 255, 0)), 2)
            cv2.putText(out, f"{classes[c - 1]} {s[m][idx]:.2f}", (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors.get(c, (0, 255, 0)), 1)
    cv2.imwrite(f"{OUT_DIR}/pred_{fname}", out)
    print("Saved", f"{OUT_DIR}/pred_{fname}")
