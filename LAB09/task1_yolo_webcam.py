import time
import cv2
from ultralytics import YOLO

# Any YOLO version works; yolov8n is small enough for CPU. Weights download automatically.
model = YOLO("yolov8n.pt")
class_names = model.names   # COCO class names

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

prev = time.time()
while True:
    ok, img = cap.read()
    if not ok:
        break

    results = model(img, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = class_names[int(box.cls[0])]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    now = time.time()
    cv2.putText(img, f"FPS: {1 / (now - prev):.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    prev = now

    cv2.imshow("YOLO Webcam - lab assets", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
