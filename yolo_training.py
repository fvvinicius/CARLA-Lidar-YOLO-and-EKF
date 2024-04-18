from ultralytics import YOLO

model = YOLO('yolov8.yaml')

# Train the model
results = model.train(data='/mnt/c/Users/vinif/Documents/Self_Driving/data.yaml', epochs=1, imgsz=1920)