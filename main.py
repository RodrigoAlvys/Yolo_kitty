from ultralytics import YOLO

photo = "./images.jpg"

model8 = YOLO("yolov8n.pt")
model26 = YOLO("yolo26n.pt")
modeltrained = YOLO("./best.pt")

resul8 = model8(photo)
resul26 = model26(photo)
resul_trained = modeltrained(photo)

for draw in resul8:
    boxes = draw.boxes
    draw.save(filename="boxes8.jpg")
for draw in resul26:
    boxes = draw.boxes
    draw.save(filename="boxes26.jpg")
for draw in resul_trained:
    boxes = draw.boxes
    draw.save(filename="boxes_trained.jpg")

