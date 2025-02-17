from ultralytics import YOLO
import torch
import torchsummary
from torchinfo import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# YOLOv8 모델 로드 (Nano 버전)
model = YOLO("yolov8n.pt")
model.to(device)

# 모델 정보 출력
model.info() # 이거랑 torchsummary랑 뭐가 다른지 확인해보자.
print('---' * 20)
summary(model.model, input_size=(1, 3, 640, 640), col_names=['input_size', 'output_size', 'num_params'])
# torchsummary.summary(model.model.model, (3, 640, 640)) # summary...

# COCO 데이터셋으로 학습 시작
model.train(data="coco.yaml", epochs=50, batch=16, imgsz=640, device = "cuda:1")

# 모델 평가 (Validation 데이터 사용)
metrics = model.val()
print(metrics)