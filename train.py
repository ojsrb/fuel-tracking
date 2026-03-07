from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(data="dataset/data.yaml", epochs=256, imgsz=640, device="cpu")

model.save("fuel.pt")