import torch
import cv2
import os
import numpy as np

# 1. YOLOv5 загварыг ачааллаж байна
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', verbose=False)

# 2. Видео эх үүсвэр (0 = камер, эсвэл видео файл)
# cap = cv2.VideoCapture("/Users/munguunuu/Documents/yolo zurguud/walking1.mov")
cap = cv2.VideoCapture("http://172.20.10.3:8081/stream")  # IP stream ашиглах бол

# 3. Камерын фреймийн өргөн -> төвийг тодорхойлох
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
center_x = frame_width // 2

# 4. Гэрлэн дохионы өнгө таних функц
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

# 5. Үндсэн real-time loop
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

        # Bounding box зурж текст нэмэх
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ➤ Гэрлэн дохионы өнгө таних
        if class_name == "traffic light":
            cropped = frame[y1:y2, x1:x2]
            color = get_dominant_color(cropped)
            voice_messages.append(f"Traffic light is {color}")

        # ➤ Замын тэмдэг бол зай, байрлал тооцоолж хэлэх
        elif "sign" in class_name or "traffic" in class_name:
            height = y2 - y1  # Bounding box-ийн өндөр
            if height > 180:
                distance = "1 meter"
            elif height > 120:
                distance = "2 meters"
            elif height > 80:
                distance = "3 meters"
            else:
                distance = "more than 4 meters"

            # Зүүн эсвэл баруун талд байгааг тогтоох
            object_center_x = (x1 + x2) // 2
            side = "left" if object_center_x < center_x else "right"
            voice_messages.append(f"{class_name} on your {side} at around {distance}")

    # Илэрсэн мэдээллийг дуугаар хэлэх
    if voice_messages:
        full_message = " | ".join(voice_messages)
        print("🗣", full_message)
        os.system(f"say '{full_message}'")  # MacOS дээр ажиллана

    # Илэрсэн зүйлтэй дүрсийг харуулах
    cv2.imshow("YOLOv5 - Video Object + Distance Detection", frame)

    # ‘q’ дарвал гарна
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. Нөөц чөлөөлөх
cap.release()
cv2.destroyAllWindows()
