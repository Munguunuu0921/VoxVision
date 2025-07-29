import torch
import cv2
import os
import numpy as np

# 🧠 YOLOv5 загварыг ачааллах
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', verbose=False)

# 🎥 Камер асаах (0 бол default webcam)
cap = cv2.VideoCapture(0)

# Дүрсний өргөнийг авах (зурагны төвийг тодорхойлох)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
center_x = frame_width // 2

# 🎨 Гэрлэн дохионы өнгө шалгах функц
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

# 🔁 Real-time loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO-р бүх обьект таних
    results = model(frame)
    names = results.names
    detections = results.xyxy[0]
    messages = []

    # Илэрсэн обьект бүрийг шалгах
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        class_name = names[int(cls)]
        x_center = (x1 + x2) // 2
        side = "left" if x_center < center_x else "right"

        # 🚦 Гэрлэн дохио бол өнгө таних
        if class_name == "traffic light":
            cropped = frame[y1:y2, x1:x2]
            color = get_dominant_color(cropped)
            messages.append(f"Traffic light is {color}")

        # 🚸 Замын тэмдэг бол шууд хэлэх
        elif "sign" in class_name or "traffic" in class_name:
            messages.append(f"Road sign detected: {class_name}")

        # 🧱 Бусад обьектууд
        else:
            messages.append(f"{class_name} on the {side}, go {'right' if side == 'left' else 'left'}")

    # 🗣 Илэрсэн мэдээллийг Mac-ийн `say` коммандаар хэлэх
    if messages:
        full_message = " | ".join(messages)
        print("👉", full_message)
        os.system(f"say '{full_message}'")

    # 🖼 Bounding box-уудтай дүрсийг харуулах
    results.render()
    cv2.imshow("Smart Assistant", results.ims[0])

    # 'q' дарвал гарах
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 🧹 Камер болон цонх хаах
cap.release()
cv2.destroyAllWindows()
