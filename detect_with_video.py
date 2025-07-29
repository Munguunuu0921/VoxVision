import torch
import cv2
import os
import numpy as np

# YOLOv5 загварыг ачааллах
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', verbose=False)

# Видео эх үүсвэр (0 бол камер, эсвэл 'video.mp4' гэж бичнэ)
cap = cv2.VideoCapture("/Users/munguunuu/Documents/yolo zurguud/walking2.mov")

if not cap.isOpened():
    print("Видео нээж чадсангүй")
    exit()

# Видео эх үүсвэр (IP camera stream)
# cap = cv2.VideoCapture("http://172.20.10.4:8081/stream")

# if not cap.isOpened():
#     print("Камераас стрим уншиж чадсангүй")
#     exit()


# Frame өргөн (зурагны төвийг бодохоор)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
center_x = frame_width // 2

# Гэрлэн дохионы өнгө таних функц
def get_dominant_color(cropped_img):
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    avg_color = hsv.mean(axis=0).mean(axis=0)
    h = avg_color[0]
    if 0 <= h <= 25:
        return 'red'
    elif 26 <= h <= 35:
        return 'yellow'
    elif 36 <= h <= 85:
        return 'green'
    else:
        return 'unknown'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    names = results.names
    detections = results.xyxy[0]
    voice_messages = []

    # Илэрсэн бүх обьектийг шалгах
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        class_name = names[class_id]
        label = f"{class_name}"

        # Bounding box зурна
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Зөвхөн тодорхой обьектоор дуу хоолой дамжуулах
        if class_name == "traffic light":
            cropped = frame[y1:y2, x1:x2]
            color = get_dominant_color(cropped)
            voice_messages.append(f"Traffic light is {color}")

        elif "sign" in class_name or "traffic" in class_name:
            voice_messages.append(f"Road sign detected: {class_name}")

    # Хэрвээ зөвхөн тодорхой зүйл илэрсэн бол хэлнэ
    if voice_messages:
        full_message = " | ".join(voice_messages)
        print("🗣", full_message)
        os.system(f"say '{full_message}'")  # Mac дээр ажиллана

    # Илэрсэн обьекттой зураг харуулна
    cv2.imshow("YOLOv5 - Video Selective Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
