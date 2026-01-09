from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11x-pose.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="coco-pose.yaml", epochs=100, imgsz=640, save=True, pretrained=True, batch=8)
