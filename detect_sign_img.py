import torch
import cv2
import os
import numpy as np


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', verbose=False)


image_path = "img9.png" 
image = cv2.imread(image_path)

if image is None:
    print("❌ Зураг олдсонгүй. Замыг шалгана уу.")
    exit()

# Төвийг тодорхойлох
frame_width = image.shape[1]
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

# 🧠 Зурагнаас бүх обьект илрүүлэх
results = model(image)
names = results.names
detections = results.xyxy[0]
messages = []

# 🔁 Илэрсэн обьект бүрийг шалгах
for *box, conf, cls in detections:
    x1, y1, x2, y2 = map(int, box)
    class_name = names[int(cls)]
    x_center = (x1 + x2) // 2
    side = "left" if x_center < center_x else "right"

    # 🚦 Гэрлэн дохио бол өнгийг нь хэлэх
    if class_name == "traffic light":
        cropped = image[y1:y2, x1:x2]
        color = get_dominant_color(cropped)
        messages.append(f"Traffic light is {color}")

    # 🚸 Замын тэмдэг (road sign) бол шууд хэлэх
    elif "sign" in class_name or "traffic" in class_name:
        messages.append(f"Road sign detected: {class_name}")

    # 🧱 Бусад обьектуудыг байрлалаар хэлэх
    else:
        messages.append(f"{class_name} on the {side}, go {'right' if side == 'left' else 'left'}")

# 🗣 Танигдсан зүйлсийг Mac-ийн `say` командыг ашиглан хэлэх
if messages:
    full_message = " | ".join(messages)
    print("🗣", full_message)
    os.system(f"say '{full_message}'")

# 🖼 Илэрсэн обьектуудтай зургыг харуулах
results.render()
cv2.imshow("YOLOv5 - Image Detection", results.ims[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
