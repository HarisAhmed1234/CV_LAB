import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Dataset layout: materials/wood/*.jpg, materials/metal/*.jpg, materials/fabric/*.jpg
DATA_DIR = "materials"
CLASSES = ["wood", "metal", "fabric"]
IMG_SIZE = (200, 200)

RADIUS = 1
N_POINTS = 8 * RADIUS


def lbp_features(gray):
    """Uniform LBP histogram (normalised)."""
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3),
                           range=(0, N_POINTS + 2))
    hist = hist.astype("float")
    return hist / (hist.sum() + 1e-6)


def glcm_features(gray):
    """Contrast, energy, homogeneity and correlation from the GLCM
    at four orientations, averaged."""
    glcm = graycomatrix(gray, distances=[1, 2],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)
    props = ["contrast", "energy", "homogeneity", "correlation"]
    return np.array([graycoprops(glcm, p).mean() for p in props])


def extract(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    return np.concatenate([lbp_features(img), glcm_features(img)])


X, y = [], []
for label, cls in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, cls)
    for name in os.listdir(folder):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            X.append(extract(os.path.join(folder, name)))
            y.append(label)

X, y = np.array(X), np.array(y)
print("Samples:", X.shape[0], "| Feature length:", X.shape[1])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)

model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
model.fit(X_tr, y_tr)
pred = model.predict(X_te)

print(classification_report(y_te, pred, target_names=CLASSES))
print("Confusion matrix:\n", confusion_matrix(y_te, pred))
