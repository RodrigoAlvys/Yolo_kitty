from ultralytics import YOLO
from huggingface_hub import snapshot_download
import os

dataset_dir = snapshot_download(
    repo_id="sankalpa1998/cat-dog-yolo-dataset",
    repo_type="dataset",
    local_dir="/content/cat-dog-yolo"
)

yaml_content = f"""
path: {dataset_dir}
train: images
val: images

names:
  0: Cat
  1: Dog
"""
with open(os.path.join(dataset_dir, "data.yaml"), "w") as f:
    _=f.write(yaml_content)

model = YOLO("yolov8n.pt")
model.train(data=os.path.join(dataset_dir, "data.yaml"), epochs=50, imgsz=640)

