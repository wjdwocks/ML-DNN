import json
import os
# JSON 파일 경로
json_file_path = "~/data/aihub/pet_data/labels/val/A_10_BEA_CM_20230131_10_105163_01.json"
json_file = os.path.expanduser("~/data/aihub/pet_data/labels/val/A_10_BEA_CM_20230131_10_105163_01.json")

# JSON 파일 열기
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)  # JSON을 Python 딕셔너리로 변환

species_value = data["annotations"]["yolo_format"]["species"]
print(f"Species: {species_value}")
