# YOLOv1의 동작 원리

YOLO(You Only Look Once)는 이미지에서 객체를 한 번의 Forward Pass로 탐지하는 실시간 Object Detection 모델이다.  
YOLOv1의 동작 과정은 **Prediction(예측)**과 **Training(학습)** 두 단계로 나뉜다.

---

## 1️⃣ YOLO의 동작 원리 (Prediction)
YOLO는 학습된 모델을 사용하여 새로운 이미지에서 객체를 탐지한다.

### 📌 **Prediction 과정**
1. **이미지를 S x S 크기의 겹치지 않는 Grid로 나눈다.**
2. **각 Grid Cell에서 사전에 학습된 가중치로 Bounding Box를 생성한다.**
3. **각 Bounding Box마다 Confidence Score(P(Obect) x IoU)를 계산한다.**
4. **계산된 Confidence Score가 Threshold를 넘지 못하면 제거한다.**
5. **NMS(Non-Maximum Suppression)을 적용하여 하나의 객체마다 가장 높은 Confidence Score를 가진 Bounding Box만 가지도록 중복을 제거한다.**
6. **그렇게 생성된 Bounding Box와 해당 객체에 대한 label값을 Predict 하게 된다.**

---

## 2️⃣ YOLO의 동작 원리 (Training)
YOLO는 학습(Training) 과정에서 모델의 가중치를 조정하여 최적의 Bounding Box를 예측할 수 있도록 한다.

### 📌 **Training 과정**
1. **각 Epoch마다 YOLO는 이미지를 S x S 크기의 겹치지 않는 Grid로 나눈다.**
2. **각 Grid Cell이 포함하는 객체의 중심을 기준으로 탐지 확률 P(Object)를 계산하고, 가중치 Parameter를 기반으로 Bounding Box를 생성한다.**  
   - ⚠️ **YOLOv2부터는 Bounding Box가 Anchor Box로, 여러 크기로 존재한다. (이전에는 랜덤 크기.)**
3. **모든 Bounding Box의 Parameter 가중치와 P(Object)를 예측하는 Parameter들은 Loss 값을 기반으로 업데이트된다.**
4. **Loss 값은 세 가지 요소로 구성된다.**
   - **Localization Loss**: Bounding Box의 위치x, y, w, h가 GT와 가까워지도록 조정하는 손실
   - **Objectness Loss**: 객체가 존재하는지 여부를 예측하는 Confidence Score에서 P(Object)의 손실
   - **Classification Loss**: 예측된 객체가 올바른 클래스로 분류되도록 하는 손실

---

## 3️⃣ 추가 공부할 개념들

### **Bounding Box란?**
- 이미지 안에서 찾고자 하는 Object의 위치를 네모 박스로 표현하는 개념.
- Object Detection 모델의 목표는 이 Bounding Box를 정확하게 예측하는 것이다.
- Bounding Box는 다음과 같은 방식으로 표현된다.
  - (x_min, y_min, x_max, y_max)
  - (x_center, y_center, width, height)  
- **YOLO에서는 (x_center, y_center, width, height) 방식**을 주로 사용한다.

### **IoU란? (Intersection over Union)**
- Ground Truth와 Predicted Box가 얼마나 겹치는지 계산하는 지표.
- 값이 1에 가까울수록 예측이 정확함을 의미한다.
- Ground Truth는 정답 label(Bounding Box)을 의미하고, Predicted Box는 모델이 예측한 Bounding Box를 의미한다.

### **NMS란? (Non-Maximum Suppression)**
- Object Detection에서 중복된 Bounding Box를 제거하는 알고리즘이다.
- Object Detection 모델은 하나의 객체에 대해 여러 개의 Bounding Box를 예측하는 경우가 많다.
- **이때, Confidence Score가 가장 높은 Bounding Box만 남기고, 나머지는 제거하는 과정이 NMS이다.**
- 즉, NMS는 모델이 최종 예측을 수행할 때 존재하는 Bounding Box들 중 하나를 선택하기 위한 것이다.  
  (⚠️ **학습 때는 적용되지 않으며, Validation 및 Inference 시에만 사용됨.**)
- **학습 때는 존재하는 모든 Bounding Box들이 GT와 IoU가 높아질수록 좋기 때문이다.**

### **Confidence Score란?**
- 특정 Bounding Box가 실제로 객체를 포함하는지를 예측하는 점수.
- Confidence Score = P(Object) x IoU(Predicted Box, GT)
  - P(Object): 해당 Grid Cell 안에 객체가 존재할 확률 (모델이 학습을 통해 예측하는 값.)
   - Training 과정 : Ground Truth에서 Grid Cell안에 객체의 중심이 존재하면 1, 아니라면 0이다.
   - Predict 과정 : CNN을 통해 특징을 추출한 후, Grid Cell 별로 P(Object)를 예측하는 출력값을 생성한다. (0~1 사이 값.)
  - IoU(Predicted Box, GT): 예측된 Bounding Box와 실제 정답(GT) 박스의 IoU
- 즉, **Confidence Score는 Bounding Box마다 가지는 점수이며, P(Object)와 IoU를 곱하여 계산된다.**


---

## 4️⃣ Anchor Box 알고리즘이란?

YOLOv2부터 도입된 **Anchor Box 알고리즘**은 **객체 크기를 더욱 정밀하게 예측**하기 위해 만들어졌다.  
YOLOv1과 YOLOv2 이후의 차이를 이해하면, Anchor Box의 필요성을 쉽게 파악할 수 있다.

### **📌 YOLOv1의 한계 (Bounding Box 직접 예측 방식)**
- YOLOv1에서는 **Grid Cell이 직접 Bounding Box의 위치와 크기 x, y, w, h 를 예측**해야 했다.
- 하지만 이 방식에서는 **한 이미지 안에 작은 객체와 큰 객체가 모두 존재하는 경우** 특정 크기의 객체만 잘 탐지되고, 다양한 크기의 객체를 탐지하기 어려웠다.
- YOLOv1에서는 **Bounding Box의 크기가 학습 초기에는 랜덤하게 설정**되기 때문에, 최적의 Box를 찾는 데 비효율적이었다.

### **📌 YOLOv2 이후, Anchor Box 개념 도입**
- **Anchor Box에서는 미리 정해진 크기와 비율의 "Anchor Box" (기준 박스)를 사용하여 Bounding Box를 초기화한다.**
- 즉, 기존에는 B = 3이더라도, 완전히 랜덤한 크기의 Bounding Box를 생성했지만,  
  **YOLOv2부터는 B = 3이라면, 각 Bounding Box들이 서로 다른 정해진 크기의 Anchor Box(예: 12×12, 24×24, 36×36)로 초기화된다.**
- 여기서 **각 Anchor Box의 크기는 YOLO 모델이 K-Means Clustering을 사용하여 자동으로 설정한다.**
- 이후 학습을 통해 Anchor Box가 GT에 맞게 조정되면서 최적화된다.

### **📌 Anchor Box 학습 과정**
1. **초기화 단계**  
   - Anchor Box 크기는 K-Means Clustering을 이용하여 자동으로 설정됨.  
   - 즉, YOLO는 학습 데이터셋을 분석하여 **가장 자주 등장하는 Bounding Box 크기들을 Anchor Box로 설정함.**  
2. **Bounding Box가 예측할 객체 선택**  
   - 각 Bounding Box는 처음 초기화된 후 **가장 적절한 클래스를 목표로 학습이 진행됨.**  
   - 즉, **각 Bounding Box는  P(Object) , Bounding Box 위치( x, y, w, h ), 클래스 확률  P(Class) 를 예측함.**  
   - GT와 **가장 IoU가 높은 Anchor Box가 해당 객체를 학습하도록 연결됨.**  
3. **GT와 연결된 Bounding Box 학습 진행**  
   - 각 GT는 **자신과 가장 IoU가 높은 Anchor Box와 연결되며**, 해당 Box가 GT의 위치와 클래스를 학습함.  
   - GT와 연결되지 않은 Bounding Box들은 **객체가 없는 것으로 간주하고, Confidence Score( P(Object) )를 낮추는 방향으로 학습됨.**  
   - 이를 **Negative Loss**라고 하며, 잘못된 Bounding Box는 Confidence Score를 0으로 조정하는 방식으로 최적화됨.

✅ **즉, YOLOv2 이후부터는 Bounding Box가 완전히 랜덤하게 설정되지 않고, Anchor Box를 기반으로 초기화된 후, 학습을 통해 최적의 크기와 위치로 조정되는 방식이다!**  

---

## 5️⃣ YOLOv3에서 달라진 점.

### **📌Darknet-53 이라는 새로운 Backbone Network 도입.**
- YOLOv2는 Darknet-19를 사용했었는데, 이보다 깊고 강력한 구조를 사용함.
- ResNet 스타일의 Residual Connections를 적용하여 더 깊은 네트워크로 성능을 향상시키면서도 연산량을 줄이는 효과를 챙겼다.
- Darknet-53은 ResNet-50 정도의 성능이면서, 속도는 더 빠르다고 한다.

### **📌Multi-Scale Prediction (FPN 구조를 도입함.)**
- YOLOv3부터는 3개의 서로 다른 Scale에서 객체를 예측한다.
- 작은 객체, 중간 크기 객체, 큰 객체를 각각 예측할 수 있도록 설계되어서 작은 객체에 대한 탐지 성능이 크게 향상됨.
- Multi-Scale Prediction의 작동 방식
   - 네트워크의 서로 다른 깊이에서 3개의 Feature Map을 추출한다.
   - 작은 scale(13 x 13), 중간 scale(26 x 26), 큰 scale(52 x 52)에서 객체를 예측한다. (크기는 예시임. input = (416, 416) 기준.)
   - 작은 객체는 해상도가 높은 Feature Map에서, 큰 객체는 해상도가 낮은 Feature Map에서 잘 감지가 되는 경향이 있다.
   - 각 Feature Map이라는 것은, Darknet-53도 Resnet처럼 여러 개의 Stage -> Block을 통하는 구조로 되어있기 때문에, 각 Stage를 통과한 Feature Map을 사용한다는 의미이다.
   - 적용 방식.
      - 즉, 처음에는 Anchor Box(를 통해 NN으로 결정된 값인) Bounding Box를 가지게 된다.
      - 그렇게, 각자 크기가 다른 Bounding Box들이 생성될 것이다.
      - 그러고, 각 Bounding Box들은 자신과 비슷한 크기의 객체를 찾으려고 할 것임. + (객체의 크기에 더 맞게 조정도 되어갈 것이다.)
      - 이 때 모든 layer에서 학습이 되기는 하지만, 각 Stage별로 작은 객체는 해상도가 높은 Feature Map(을 가진 layer), 큰 객체는 해상도가 낮은 Feature Map(을 출력하는 layer)에서 잘 detect될 가능성이 높다.
      - 이렇기 때문에, Layer가 깊어진 것 + Anchor Box 방식을 통해서 작은 크기의 객체를 더 잘 탐지하게 되었다는 것을 의미한다.
- Anchor Box와 뭐가 다른거지?
   - 완전히 다름. 
   - Multi Scale Prediction은 다양한 Feature Map을 통해서 다양한 크기의 Object를 예측하도록 하는 기술을 의미함.
   - Anchor Box는 각 Feature Map에서 예측된 Bounding Box의 크기를 조정하는 과정을 의미함.
   - 즉, Multi Scale Prediction과 Anchor Box는 서로 다른 개념이고, 같이 동작함으로써 학습 및 예측을 수행한다.


### **📌Bounding Box의 예측 방식 변경.(Logistic Regression)**
- YOLOv3는 Bounding Box의 중심 좌표(x, y)를 예측할 때 Logistic Regression을 사용한다.
- 이를 통해서 Bounding Box가 Grid Cell을 벗어나지 않도록 안정적으로 조정된다고 함.
- 예측된 x, y는 (0 ~ 1)사이 값을 갖도록 하며, 이 값은 Grid Cell 내부의 상대 좌표로 변환된다.
   - logistic regression은 sigmoid 함수를 통해 (x, y)의 값을 (0 ~ 1)범위로 제한하기 때문임..
- 왜 Bounding Box의 중심이 Grid Cell 내부에만 존재하도록 제한하는 것이 좋은가?
   - 만약 Bounding Box가 Grid Cell외부로 나간다면, 하나의 객체를 여러 Grid Cell에서 중복으로 예측하려고 할 수 있다. (매우 비효율적.)
   - 즉, 애초부터 Grid Cell을 나눈 이유가, 그 구역 내의 객체를 정확히 탐지시키기 위한것인데, 탐지하기 쉬운 물체가 Grid Cell 밖에 있다면 거기로 몰려버리게 되어서 제대로된 학습이 되지 않음.


### **📌클래스 예측 방식의 변경 (Softmax → 독립적인 Sigmoid)**
- YOLOv2에서는 Softmax를 사용하여 다중 클래스 확률을 예측했다. (하나의 Bounding Box마다 1개의 클래스)
   - 이러한 방식의 문제는 객체가 여러 클래스를 가질 가능성이 있는 경우(**자전거를 탄 사람**) 적절하지 못하다.
- 그래서 YOLOv3는 각 클래스에 대해 독립적인 Sigmoid 확률을 계산하여, Multi-label Classification이 가능해졌다.
- 그래서 각 클래스마다 Sigmoid값을 계산하여 특정 threshold를 넘으면 그 클래스가 맞다고 예측하는건가?
   - 특정 threshold 를 넘는다면, 그 클래스(label)이 그 boundingbox 내에 존재한다고 판단한다.
   - 보통 threshold는 0.5정도를 사용한다고 함. (물론 데이터셋마다 다르겠지만.)


### **📌새로운 손실 함수를 설계함.**
- YOLOv3부터는 IoU 기반의 새로운 손실 함수를 도입하였다.
- Bounding Box 예측의 정확도를 높이기 위한 IoU 기반의 좌표 손실을 개선하였다.
   - IoU : (두 Bounding Box의 교집합) / (두 Bounding Box의 합집합.) - 여기서 두 Bounding Box란, 예측한 것과 Ground Truth의 Bounding Box를 의미함.
   - GIoU (Generalized IoU) : IoU - (C - U) / C , 
      - GIoU는 IoU를 개선한 버전으로, 두 Bounding Box가 겹치지 않을 때에도 학습이 가능하도록 한다.
      - C : 두 Bounding Box를 포함하는 최소한의 영역. (최소 외접 박스)
      - U : 두 Bounding Box의 합집합.
   - DIoU (Distance IoU) : IoU - (ρ² / c²) 
      - DIoU는 중심 간 거리를 최소화하여 Bounding Box를 더 빠르게 정렬하게 해준다.
      - ρ : 두 Bounding Box 중심 간 거리.
      - c : 두 Bounding Box를 포함하는 최소 외접 박스의 대각선 길이.
   - CIoU (Complete IoU) : IoU - (ρ² / c²) - αv
      - CIoU는 IoU + 중심 거리 + Aspect Ratio까지 고려하여 가장 안정적인 Bounding Box를 조정한다. (YOLOv4, YOLOv5에서도 기본적용됨.)
      - α : 가중치
      - v : Bounding Box의 가로세로 비율 차이.

- 최소외접 박스
<img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/25.2.20/Minimum_Enclosing_Box.png" alt="최소외접박스" width="500">

---

## 🎯 **YOLO 핵심 요약**
- **이미지를 S x S 크기의 Grid로 나누고, 각 Grid Cell에서 Bounding Box를 예측하는 방식**
- **Bounding Box의 Confidence Score를 계산하여, 특정 Threshold 이상인 박스만 유지 (예측을 수행할 때에만)**
- **Loss를 통해 Bounding Box의 위치, Confidence Score, 클래스 예측을 최적화**
- **Validation 단계에서 NMS를 적용하여 최종 Bounding Box를 선택**
- **YOLOv2 이후부터는 Anchor Box를 도입하여 다양한 크기의 객체 탐지가 가능하도록 개선됨**
- **YOLO의 출력 : {S x S x (B x 5 + C)}**
   - S : 나뉘는 Grid Cell의 개수. (S x S 개의 Grid Cell로 나뉘게 된다.)
   - B : 각 Grid Cell마다 가지는 Bounding Box의 개수.
      - 각 Bounding Box에는 (x, y, w, h, P(obj)) 의 다섯 개의 값을 가지므로 (B x 5)로 표현됨.
   - C : 각 Bounding Box 마다 Class를 예측해야하기 때문에 C개의 클래스 개수도 포함하게 된다. (각 클래스 마다의 확률을 얻고, 가장 높은 것을 선택함.)

---

## ✅ **10년 뒤에도 쉽게 이해할 수 있도록!**
📌 **YOLOv1은 Bounding Box를 직접 예측하는 방식이었지만, YOLOv2 이후부터는 Anchor Box를 도입하여 더욱 정교한 탐지가 가능해졌다.**  
📌 **YOLO는 한 번의 Forward Pass로 전체 이미지에서 객체를 탐지할 수 있도록 설계된 실시간 Object Detection 모델이다.**  
📌 **이 문서를 다시 읽을 때는 YOLO 최신 버전과 비교하면서 차이점을 분석해볼 것!**


1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ 🔟
