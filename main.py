from ultralytics import YOLO

model = YOLO("41ep-16-just-drone-GPU.pt")

result = model.train(data="data.yaml", epochs=10, device="mps", batch=16)