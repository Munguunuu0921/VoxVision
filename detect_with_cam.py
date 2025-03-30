import torch
import cv2
import os

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 загвар татах
cap = cv2.VideoCapture(0)  # Mac-ийн webcam

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
        print(sentence)
        os.system(f"say '{sentence}'")  # Mac дээрх Text-to-Speech

    results.render()
    cv2.imshow("YOLO Camera Detection", results.ims[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
