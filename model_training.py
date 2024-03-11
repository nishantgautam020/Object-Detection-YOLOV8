from ultralytics import YOLO

def model_train():
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # Use the model
    model.train(data="config.yaml", epochs=3)  # train the model

model_train()