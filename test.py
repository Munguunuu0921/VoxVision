import torch
import cv2
import os
import numpy as np

# YOLOv5 загварыг TorchHub-аас ачааллаж байна
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Камер асаах
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    labels = results.xyxyn[0][:, -1].numpy()
    names = results.names
    detected = list(set([names[int(i)] for i in labels]))

    if detected:
        sentence = "Objects detected: " + ", ".join(detected)
        os.system(f"say '{sentence}'")

    results.render()
    cv2.imshow("VoxVision", results.ims[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()