import json
import os

# COCO JSON 파일 경로
json_path = "annotations/instances_train2017.json"
output_dir = "labels/train2017/"  # YOLO 라벨 저장 폴더
os.makedirs(output_dir, exist_ok=True)

# JSON 파일 열기
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    
for i in range(3):
    print(data['annotations'][i])

print('-------' * 5)
print('데이터 변환 시작 (json → yolo_txt)')

# 각 이미지별로 YOLO 라벨 생성
for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"] - 1  # YOLO는 0부터 시작
    bbox = annotation["bbox"]  # [x_min, y_min, width, height]

    # 해당 이미지 정보 찾기
    image_info = next((img for img in data["images"] if img["id"] == image_id), None)
    if not image_info:
        continue

    img_width = image_info["width"]
    img_height = image_info["height"]

    # COCO BBox → YOLO BBox 변환
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width /= img_width
    height /= img_height

    # YOLO 포맷으로 저장
    label_path = os.path.join(output_dir, f"{image_id:012d}.txt")
    with open(label_path, "a") as label_file:
        label_file.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

print("✅ COCO JSON → YOLO TXT 변환 완료!")
