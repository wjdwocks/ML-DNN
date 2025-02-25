from ultralytics import YOLO
import torch
import torchsummary
from torchinfo import summary

torch.cuda.set_device(1)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# YOLOv8 모델 로드 (Nano 버전)
model = YOLO("runs/detect/train27/weights/best.pt")
model.to(device)

# 모델 정보 출력
model.info() # 이거랑 torchsummary랑 뭐가 다른지 확인해보자.
print('---' * 20)
summary(model.model, input_size=(1, 3, 640, 640), col_names=['input_size', 'output_size', 'num_params'])

# COCO 데이터셋으로 학습 시작
model.train(data="coco.yaml", epochs=100, patience = 50, lr0 = 0.01, batch=32, imgsz=640, device = "cuda:1",  weight_decay=0.001, dropout=0.1, mosaic=0.5, mixup=0.2)

# 모델 평가 (Validation 데이터 사용)
metrics = model.val(data="coco.yaml", save_json=True, shuffle=True)
print(metrics)