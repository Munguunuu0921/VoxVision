import torch
import cv2
import os

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
cap = cv2.VideoCapture("http://192.168.1.123:81/stream")  # IP stream

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream тасарсан")
        break

    results = model(frame)
    labels = results.xyxyn[0][:, -1].numpy()
    names = results.names
    detected = list(set([names[int(i)] for i in labels]))

    if detected:
        message = "Урд талд: " + ", ".join(detected)
        print(message)
        os.system(f"say '{message}'")  # macOS-д Text-to-Speech

    results.render()
    cv2.imshow("Helmet IP Detection", results.ims[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
