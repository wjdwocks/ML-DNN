## U-Net 공부
- U-Net의 등장 이유
    * 기존 CNN은 분류 작업에는 강했지만, 픽셀 단위로 의미를 파악해야 하는 작업(Segmentation)에는 성능이 약했다.
    * 기본 CNN은 다운 샘플링(Pooling, Stride) 때문에 해상도가 감솧고, 위치 정보가 소실되기 때문에.
    * 예를 들어 고양이 이미지가 있을 때 고양이가 있는지는 잘 맞추지만, 고양이의 경계를 구분하기는 어려웠다.
    * Fully Convolutional Network에서는 해상도 복원이 불충분하고, 저해상도 Feature로 복원하기에 경계가 흐릿했다.
    * 즉, 해상도와 위치 정보를 손실 없이 유지하면서, CNN의 추상화 능력을 함께 사용할 수 있도록 U-Net이 설계됨.

- U-Net의 구조
    * Encoder
        - 일반적은 CNN과 비슷하다.
        - Convolution → ReLU → MaxPooling을 반복함.
        - 입력 이미지에서 고수준의 Feature를 추출하는 것이 목표.
    * Decoder (Expanding Path)
        - ConvTranspose를 통한 UpSampling → Convolution
        - Feature를 점점 다시 원래 크기로 키운다.
        - 해당 객체가 어디에 있는지를 예측함.
    * Skip Connection
        - Encoder의 중간 Feature Map을 Decoder에 연결해준다.
        - Encoder의 Feature Map을 Decoder로 채널 차원으로 concat 해줌.
        - 이 Skip Connection을 통해 Decoder는 Encoder에서 잃어버린 공간(위치) 정보를 복원하여 학습이 가능하다.

- U-Net의 작동 방식
    * Encoder
        - 입력 : 예를 들어 224 x 224 x 3 (Image Net)
        - Conv 블록 + MaxPooling을 반복하며 해상도를 줄이고, 채널 수를 늘려간다.
        - (224×224×3) → (112×112×64) → (56×56×128) → (28×28×256) → (14×14×512)
    * BottleNeck
        - 해상도는 유지하면서(14 x 14), 채널 수만 증가시켜 더욱 추상적인 Feature를 추출한다.
        - Conv Layer를 통해 가장 압축되고 의미적인 정보를 생성한다. (14x14x512) → (14x14x1024)
    * Decoder
        - Encoder에서 압축된 정보를 가지고, 원본 이미지로 되돌리는 방식으로 학습된다.
        - ConvTrnaspose 와 같은 방식을 통해서 업스케일링 하게 됨.
        - 이 때 Skip Connection을 통해서 Encoder의 각 Layer의 결과로 나온 Feature Map들이 Decoder의 각 Layer의 입력으로 함께 들어가서 해상도가 낮아지면서 잃어가던 위치 정보를 그대로 넘겨주어 Decoder가 위치 정보를 함께 학습할 수 있도록 해준다.
        - 즉, 각 Layer의 출력(입력)에 Encoder의 Feature Map이 Concat되고, 다시 Conv Layer를 통해 원래의 크기로 맞춰주면, 그 Feature에는 위치 정보도 포함되게 되는것임.
        - (14 x 14 x 1024) -> (28 x 28 x 512) + (28 x 28 x 256 - Encoder의 Feature Map) -> (28 x 28 x 512) -> (56 x 56 x 256) + (56 x 56 x 128) -> (56 x 56 x 256) -> (112 x 112 x 128) + (112 x 112 x 64) -> (112 x 112 x 128) -> (224 x 224 x 32)
    * 출력
        - 최종적으로 Conv Layer를 통해 n개의 채널로 바꿔서 원하는 Task에 맞게 출력을 맞춰준다.
        - Segmentation이라면 (224 x 224 x n)으로 각 픽셀마다 해당하는 Class를 예측해서 CrossEntropy를 계산하도록 학습을 함.
        - Diffusion Model이었다면, (224 x 224 x 3)으로 원본 입력 채널과 맞춰줘서 얼마나 똑같은지에 대한 Loss를 학습하게 함.

- U-Net이 쓰이는 곳
    * Segmentation
        - 원래가 의료 영상 Segmentatoin을 위해 2015년에 처음 제안됨.
        - 이후 일반적인 Sementic Semgentation의 baseline 구조로 자리잡았다고 한다.
    * Diffusion Model의 Backbone
        - 대부분의 Diffusion Model에서 U-Net 구조가 Denoising Model로 사용되고 있다고 함.
    * 왜 잘쓰임?
        - Skip Connection을 통해 위치 정보가 Decoder까지 보존이 되어서 좋다.
        - Down -> Up 구조 덕분에 복원에 강하다
        - 출력 채널 수, 조건 삽입 등 구조 변경이 쉬움 ???

- 어떻게 Condition을 넣어줌?
    * Concatenation
        - 원하는 조건을 이미지처럼 Reshape해서 x_t와 채널 차원으로 concat해준다.
        - 예를들어서 학습한 프롬프트를 저렇게 Reshape해서 concat해줌.
        - 아니면 segmentation mask같은거도
    * Cross-Attention
    * Adaptive Normalization