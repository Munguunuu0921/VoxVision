import torch
import cv2
import os

model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/vision_assist/weights/best.pt')
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
        sentence = "Урд талд: " + ", ".join(detected)
        os.system(f"say '{sentence}'")

    results.render()
    cv2.imshow("Custom YOLO", results.ims[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
