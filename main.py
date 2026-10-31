from ultralytics import YOLO

model8 = YOLO("yolov8n.pt")
model26 = YOLO("yolo26n.pt")

resul8 = model8("./stray-cats.jpg")
resul26 = model26("./stray-cats.jpg")

for draw in resul8:
    boxes = draw.boxes
    draw.save(filename="boxes8.jpg")
for draw in resul26:
    boxes = draw.boxes
    draw.save(filename="boxes26.jpg")

