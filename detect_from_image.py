import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 🧠 YOLOv5 загвар ачааллах
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', verbose=False)

# 📷 Зураг унших
image_path = "img1.png"  # Зурагны нэрээ тохируулна
image = cv2.imread(image_path)

if image is None:
    print("❌ Зураг олдсонгүй. Замыг шалгана уу.")
    exit()

# 📍 Төвийг тодорхойлох
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

# 🧠 YOLO-р таних
results = model(image)
names = results.names
detections = results.xyxy[0]

messages = []
detected_labels = []
confidences = []

# 🔍 Илэрсэн обьект бүрийг шалгах
for *box, conf, cls in detections:
    x1, y1, x2, y2 = map(int, box)
    class_name = names[int(cls)]
    detected_labels.append(class_name)
    confidences.append(float(conf))

    x_center = (x1 + x2) // 2
    side = "left" if x_center < center_x else "right"

    # 🚦 Гэрлэн дохио таних
    if class_name == "traffic light":
        cropped = image[y1:y2, x1:x2]
        color = get_dominant_color(cropped)
        messages.append(f"Traffic light is {color}")

    # 🚸 Замын тэмдэг
    elif "sign" in class_name or "traffic" in class_name:
        messages.append(f"Road sign detected: {class_name}")

    # 🧱 Бусад
    else:
        messages.append(f"{class_name} on the {side}, go {'right' if side == 'left' else 'left'}")

# 🔊 Mac дээр хэлэх
if messages:
    final_msg = " | ".join(messages)
    print("🗣", final_msg)
    os.system(f"say '{final_msg}'")

# 🖼 Илэрсэн обьектуудыг зураг дээр харуулах
results.render()
cv2.imshow("YOLOv5 - Object Detection", results.ims[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

# 📊 График 1: Обьект бүрийн тоо
label_counts = Counter(detected_labels)
plt.figure(figsize=(10, 5))
plt.bar(label_counts.keys(), label_counts.values(), color='skyblue', edgecolor='black')
plt.title("Detected Object Count")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 📈 График 2: Confidence тархалт
plt.figure(figsize=(8, 4))
plt.hist(confidences, bins=10, color='lightgreen', edgecolor='black')
plt.title("Confidence Score Distribution")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()
