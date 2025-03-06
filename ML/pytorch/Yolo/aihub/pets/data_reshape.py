import os
import json
from PIL import Image
from glob import glob


print("스크립트 실행됨!")


# JSON 및 이미지 경로 설정
json_dir = os.path.expanduser("~/data/aihub/pet_data/labels/val/")
image_dir = os.path.expanduser("~/data/aihub/pet_data/images/val/")

# species 매핑 (10=dog, 20=cat)
species_map = {
    "10": "dog",
    "20": "cat"
}

# 모든 JSON 파일 가져오기
json_files = glob(os.path.join(json_dir, "*.json"))

print(f'json_files의 개수 : {len(json_files)}')

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 이미지 파일명 가져오기
    image_filename = data["annotations"]["image-id"]
    image_path = os.path.join(image_dir, image_filename)

    # 2. 실제 이미지 크기 확인 (Pillow 사용)
    if not os.path.exists(image_path):
        print(f"[Warning] Image not found: {image_path}")
        continue
    
    with Image.open(image_path) as img:
        img_width, img_height = img.size  # 실제 이미지 크기

    # 3. species (반려견/반려묘) 가져오기
    species_code = data["metadata"]["id"]["species"]  # "10" or "20"
    species = species_map.get(species_code, "unknown")  # 기본값 "unknown"

    # 4. 바운딩 박스 좌표 가져오기
    annotation = data["annotations"]["label"]
    points = annotation["points"]  # [[x_min, y_min], [x_max, y_max]]

    x_min, y_min = points[0]
    x_max, y_max = points[1]

    # 5. YOLO 형식으로 정규화 (0~1 사이 값)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    x_center = x_min + bbox_width / 2.0
    y_center = y_min + bbox_height / 2.0

    x_center /= img_width
    y_center /= img_height
    bbox_width /= img_width
    bbox_height /= img_height

    # 6. 기존 JSON 파일을 덮어쓰기 위해 YOLO 변환된 데이터를 추가
    data["annotations"]["yolo_format"] = {
        "species": species,
        "bbox": [x_center, y_center, bbox_width, bbox_height]
    }

    # 7. JSON 파일 다시 저장 (덮어쓰기)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[Updated] {json_file} with YOLO format.")
