import os
from PIL import Image
import numpy as np

image_dir = os.path.expanduser("~/data/aihub/pet_data/images/train")
image_sizes = []

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    if img_name.endswith(".jpg") or img_name.endswith(".png"):  # 이미지 파일 필터
        with Image.open(img_path) as img:
            image_sizes.append(img.size)  # (width, height)

# 평균 이미지 크기 계산
widths, heights = zip(*image_sizes)
avg_width = int(np.mean(widths))
avg_height = int(np.mean(heights))

print(f"Dataset 평균 크기: {avg_width}x{avg_height}")
