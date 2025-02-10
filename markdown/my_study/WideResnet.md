# WideResNet의 동작 원리

WideResNet(Wide Residual Network)은 ResNet을 확장한 모델로, 기존 ResNet보다 넓은 네트워크 구조를 사용하여 성능을 향상시킨다.  
특히, **WideResNet_sp**는 SPKD(Similarity-Preserving Knowledge Distillation)에 특화된 모델로, WideResNet을 기반으로 설계되었다.  
WideResNet의 **구조, Residual Block의 반복 횟수, Skip Connection, Depth 계산 방식**을 설명한다.

---

## 1️⃣ WideResNet의 모델 개요

- **WideResNet**은 기본적으로 **초기 Convolution Layer**, 이후 **3개의 Residual Stage**로 구성된다.
- 각 Residual Stage에는 여러 개의 **Residual Block**이 포함되며,  
  마지막에는 **Global Average Pooling(GAP)과 Fully Connected(FC) Layer**를 거쳐 분류 결과를 출력한다.
- **WideResNet_sp**는 SPKD 적용 시 사용하도록 설계된 특수한 WideResNet 구조이다.

---

## 2️⃣ 초기 Convolution Layer 및 Feature 확장

- 네트워크는 **conv1 레이어**를 통해 입력 이미지를 먼저 처리한다.
- **conv1 이후, 첫 번째 Residual Stage에 들어가기 전에 출력 채널 수를 16(또는 k에 따른 16×k)으로 확장**하여 후속 Stage에서 일관된 채널 수를 유지할 수 있도록 한다.

---

## 3️⃣ Residual Stage 구성과 Residual Block 반복 횟수

- 전체 Residual Stage는 **3개**로 구성된다.
- 각 Stage에 포함될 **Residual Block의 개수 N은 (d - 4) / 6 으로 계산된다.**
  - 여기서 **d**는 전체 Depth를 의미한다.
  - 예를 들어, **WRN16-3**에서는 d = 16, k = 3, 그리고 strides = [1, 1, 2, 2]로 설정된다.
  - **6으로 나누는 이유**는, 각 Residual Block 내에 **2개의 Convolution Layer**가 있으며,  
    이 블록이 **3개의 Stage**에 걸쳐 반복되기 때문이다.

---

## 4️⃣ Residual Stage의 특징 및 Skip Connection 처리

### **첫 번째 Residual Stage**
- **모든 Residual Block의 stride가 1**로 설정되어, 입력 Feature Map의 크기를 유지한다.
- 입력과 출력의 크기가 동일하므로 Skip Connection 시 원본 입력 x를 그대로 더할 수 있다.
- 단, 채널 수가 달라질 경우 **1×1 Convolution(conv_inp)**을 사용하여 채널 수를 맞춘다.  
  (WRN16-3처럼 k값이 1이 아닌 경우)

### **두 번째 및 세 번째 Residual Stage**
- **각 Stage의 첫 번째 Residual Block에서는 stride가 2**로 설정되어,  
  입력 Feature Map의 크기를 절반으로 줄인다.
- 이때 Skip Connection을 위해 원본 입력 x의 크기와 채널 수가 변경된 출력 x1과 일치해야 하므로,  
  **1×1 Convolution(conv_inp)**을 사용하여 x의 크기(및 채널 수)를 변환한다.

---

## 5️⃣ Depth 계산에 포함되는 구성 요소

- **WideResNet의 Depth d는 학습 가능한 가중치를 가진 Convolution Layer와 Fully Connected Layer만 포함하여 계산된다.**
- **BatchNorm, ReLU, Pooling** 등은 가중치가 없거나 단순한 연산이므로 Depth 계산에서 제외된다.

### **Depth 계산 공식**
- d = 1 (conv1) + 6N (Residual Blocks 내 2개의 Conv per Block, 3 Stage) + 1 (FC)


- 예를 들어, **WRN16-3에서는 1 + 6N + 1 = 16이 되어야 한다.**
- 첫 번째 Residual Stage에서는 입력 크기가 변하지 않으므로 **subsample_input = False**가 적용된다.

---

## 6️⃣ WRN161 vs. WRN163의 차이 및 첫 번째 Stage 처리

- **WRN161과 WRN163**는 **첫 번째 Residual Stage에서 출력 채널 수를 늘릴지(increase_filters 적용 여부)에 따라 구분된다.**
  - **WRN161**: 첫 번째 Stage에서 채널 수를 그대로 유지 → increase_filters = False
  - **WRN163**: 첫 번째 Stage에서도 채널 수를 확장 → increase_filters = True
- 이 차이에 따라 **Skip Connection 시 1×1 Convolution(conv_inp)의 적용 여부**가 달라지며,  
  이는 최종 모델의 Depth와 Feature Representation에도 영향을 미친다.

---

## 7️⃣ Skip Connection과 conv_inp(x)의 필요성

- **Skip Connection (x + x1)을 수행하기 위해서는 x와 x1의 크기와 채널 수가 동일해야 한다.**
- Residual Block 내에서 **Downsampling(stride=2)나 채널 증가가 발생하면** x와 x1의 shape가 달라지므로,  
  **1×1 Convolution(conv_inp)**을 사용하여 x를 변환해 두 텐서의 shape를 맞춘다.

### **Skip Connection이 필요한 경우**
1. **크기(Spatial Dimension)가 다른 경우:**  
   - 예) x의 shape이 (B, C, 32, 32)인데, x1이 (B, C, 16, 16)인 경우
2. **채널 수가 다른 경우:**  
   - 예) x의 채널이 16인데, x1의 채널이 32인 경우

---

## 8️⃣ 요약 및 결론

1. **WideResNet**은 기존 ResNet보다 넓은 구조를 사용하며, 초기 Convolution Layer와 3개의 Residual Stage, GAP 및 FC Layer로 구성된다.
2. **Residual Block의 반복 횟수 N**은 (d - 4) / 6 으로 계산되며, 각 Block 내에 2개의 Conv Layer가 포함된다.
3. **첫 번째 Residual Stage**에서는 Feature Map의 크기를 유지하기 위해 stride가 1로 설정되며,  
   채널 수가 변할 경우에만 **1×1 Convolution(conv_inp)**를 사용하여 채널을 맞춘다.
4. **두 번째 및 세 번째 Residual Stage**에서는 첫 번째 Block에서 stride가 2로 설정되어 Downsampling을 수행하며,  
   이때 conv_inp를 사용해 입력 x를 변환하여 Skip Connection을 올바르게 수행한다.
5. **Depth 계산**은 학습 가능한 가중치를 가진 Conv Layer와 FC Layer만 포함하며,  
   **BatchNorm, ReLU, Pooling 등은 제외된다.**
6. **WRN161 vs. WRN163**는 첫 번째 Residual Stage에서 채널 수 확장의 여부에 따라 구분되며,  
   이로 인해 Skip Connection을 위한 conv_inp의 적용 방식이 달라진다.
7. **Skip Connection (x + x1)을 올바르게 수행하기 위해서는 두 텐서의 shape가 동일해야 하며, conv_inp(x)는 이를 맞추기 위한 핵심 역할을 한다.**
