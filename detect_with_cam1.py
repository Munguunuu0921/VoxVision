import torch
import cv2
import os
import numpy as np

# 1. YOLOv5 загвар татаж ачааллаж байна
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 2. Камер асааж байна (0 = default webcam)
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://172.20.10.3:8081/stream")

# 3. Камерын зургийн өргөнийг авч байна (зурагны төвийг бодохоор)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
center_x = frame_width // 2

# 4. Гэрлэн дохионы дундаж өнгийг шалгах функц
def get_dominant_color(cropped_img):
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)  # RGB-г HSV болгон хувиргах
    avg_color = hsv.mean(axis=0).mean(axis=0)           # Зургийн дундаж өнгө (Hue)ssss

    h = avg_color[0]

    # HSV өнгөний утгаар шалгана
    if 0 <= h <= 25:     # Red (~0–25 hue)
        return 'red'
    elif 26 <= h <= 35:  # Yellow (~26–35 hue)
        return 'yellow'
    elif 36 <= h <= 85:  # Green (~36–85 hue)
        return 'green'
    else:
        return 'unknown'

# 5. Үндсэн real-time loop эхэлж байна
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)           # 6. YOLO-р объект таньж байна
    names = results.names            # 7. Классуудын нэрс
    detections = results.xyxy[0]     # 8. Илэрсэн объектуудын координат

    directions = []

    # 9. Илэрсэн объект бүрийг шалгах
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        class_name = names[int(cls)]

        # 10. Хэрвээ "traffic light" бол өнгийг нь шалгах
        if class_name == "traffic light":
            cropped = frame[y1:y2, x1:x2]
            color = get_dominant_color(cropped)

            if color == 'red':
                directions.append("Red light ahead, please stop.")
            elif color == 'green':
                directions.append("Green light ahead, you may go.")
            elif color == 'yellow':
                directions.append("Yellow light ahead, be cautious.")
            else:
                directions.append("Traffic light detected, color unclear.")
        
        # 11. Бусад объект (машин, хог гэх мэт)
        else:
            x_center = (x1 + x2) // 2
            side = "left" if x_center < center_x else "right"
            directions.append(f"{class_name} on the {side}, go {'right' if side == 'left' else 'left'}")

    # 12. Илэрсэн зүйл байвал дуугаар хэлнэ
    if directions:
        message = "Ahead: " + " | ".join(directions)
        print(message)
        os.system(f"say -v Alex '{message}'")

    # 13. Илэрсэн зураг дээр bounding box зурж харуулна
    results.render()
    cv2.imshow("YOLO Detection", results.ims[0])

    # 14. 'q' дарвал гарна
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 15. Камер болон цонхыг хааж байна
cap.release()
cv2.destroyAllWindows()