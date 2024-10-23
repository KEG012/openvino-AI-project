from ultralytics import YOLO
import torch

model = YOLO("yolo11n-cls.pt")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_results = model.train(
    data = "/home/keg/workspace/openvino-AI-project",
    epochs = 50,
    imgsz = 640,
    device = device
)

# Evaluate model performance on the validation set
metrics = model.val()
print(metrics)

# Perform object detection on an image
results = model("./strawberry.jpg")

if results:
    results[0].show()
else:
    print("Can't detect")

model.save("10_color_classification_model.pt")
print("save the model")