import torch
import cv2
import os
import easyocr
import numpy as np

# 🧠 YOLOv5 загварыг ачааллах
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', verbose=False)

# 🧠 EasyOCR инициализ хийх
reader = easyocr.Reader(['en'], gpu=False)

# 🖼️ Зураг ачааллах (зурагтай файлын нэрээ энд бичнэ)
image_path = "yellow.png"
image = cv2.imread(image_path)
if image is None:
    print("Зураг олдсонгүй:", image_path)
    exit()

# 📍 Камер шиг төвийг бодохын тулд зурган дээрх өргөн
frame_width = image.shape[1]
center_x = frame_width // 2

# 🎨 Гэрлэн дохионы өнгө таних функц
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

# 📦 Обьектуудыг YOLO ашиглан таних
results = model(image)
names = results.names
detections = results.xyxy[0]

messages = []

# 🔍 Обьект бүрээр давталт
for *box, conf, cls in detections:
    x1, y1, x2, y2 = map(int, box)
    class_name = names[int(cls)]
    x_center = (x1 + x2) // 2
    side = "left" if x_center < center_x else "right"

    # 🚦 Гэрлэн дохионы өнгө шалгах
    if class_name == "traffic light":
        cropped = image[y1:y2, x1:x2]
        color = get_dominant_color(cropped)
        if color == 'red':
            messages.append("Red light ahead, please stop.")
        elif color == 'green':
            messages.append("Green light ahead, you may go.")
        elif color == 'yellow':
            messages.append("Yellow light ahead, be cautious.")
        else:
            messages.append("Traffic light detected, unclear color.")

    # 🚌 Автобус илэрвэл OCR хийх
    elif class_name == "bus":
        cropped_bus = image[y1:y2, x1:x2]
        ocr_results = reader.readtext(cropped_bus)
        if ocr_results:
            text = ocr_results[0][1].replace(" ", "").upper()
            messages.append(f"Bus detected, number: {text}")
            os.system(f"say 'Bus number {text}'")

    # 🧱 Бусад обьект
    else:
        messages.append(f"{class_name} on the {side}, go {'right' if side == 'left' else 'left'}")

# 🗣 Илэрсэн зүйлсийг Mac-ийн `say` ашиглан хэлэх
if messages:
    output = " | ".join(messages)
    print("👉", output)
    os.system(f"say '{output}'")

# 🖼 Илэрсэн обьектуудтай зураг харуулах
results.render()
cv2.imshow("Image Detection", results.ims[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
