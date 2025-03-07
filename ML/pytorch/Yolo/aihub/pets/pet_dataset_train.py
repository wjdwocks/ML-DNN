from ultralytics import YOLO
import torch
from torchinfo import summary

torch.cuda.set_device(1)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')
model.to(device)

# 모델 정보 출력
model.info()

print('---' * 20)

# 모델 구조 출력 (input_size 반영)
summary(model.model, input_size=(1, 3, 640, 640), col_names=['input_size', 'output_size', 'num_params'])

# YOLO 학습 실행
model.train(
    data="pet_dataset.yaml",
    epochs=100, 
    patience=50,
    lr0=0.01, 
    batch=32, 
    imgsz=640,  # 너비는 자동 조정됨 (720x640)
    device="cuda:1",  
    weight_decay=0.001, 
    dropout=0.1, 
    mosaic=0.5, 
    mixup=0.2
)

# 모델 평가
metrics = model.val(data="pet_dataset.yaml", save_json=True, shuffle=True)
print(metrics)
