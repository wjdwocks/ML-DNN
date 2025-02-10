# YOLOv1의 동작 원리

YOLO(You Only Look Once)는 이미지에서 객체를 한 번의 Forward Pass로 탐지하는 실시간 Object Detection 모델이다.  
YOLOv1의 동작 과정은 **Prediction(예측)**과 **Training(학습)** 두 단계로 나뉜다.

---

## 1️⃣ YOLO의 동작 원리 (Prediction)
YOLO는 학습된 모델을 사용하여 새로운 이미지에서 객체를 탐지한다.

### 📌 **Prediction 과정**
1. **이미지를 S x S 크기의 겹치지 않는 Grid로 나눈다.**
2. **각 Grid Cell이 포함하는 객체의 중심을 기준으로 탐지 확률 P(Object)을 계산하고, 가중치(Weight) Parameter를 기반으로 Bounding Box를 생성한다.**
3. **각 Bounding Box의 Confidence Score를 계산한다.**  
   - Confidence Score = P(Object) x IoU(예측된 박스, GT)
4. **Confidence Score가 특정 Threshold를 넘는 Bounding Box만 남긴다.**  
   - 단, **학습(Training) 때는 Threshold를 적용하지 않으며, 모든 Bounding Box를 유지한다.**
5. **Threshold를 넘는 Bounding Box들은 Loss 값을 최소화하는 방향으로 조정되며, IoU가 높아지는 방향으로 학습된다.**
6. **Bounding Box는 Grid Cell 내부에만 한정되지 않고, 조정될 수 있다.**
7. **Validation 단계에서 NMS(Non-Maximum Suppression)를 적용하여 중복된 Bounding Box를 제거한다.**  
   - **하나의 객체에 여러 개의 Bounding Box가 예측될 수 있기 때문에, 가장 Confidence Score가 높은 Bounding Box 하나만 남긴다.**
   - 최종적으로, Confidence Score가 가장 높은 Bounding Box가 선택되어 객체의 위치를 예측한다.

---

## 2️⃣ YOLO의 동작 원리 (Training)
YOLO는 학습(Training) 과정에서 모델의 가중치를 조정하여 최적의 Bounding Box를 예측할 수 있도록 한다.

### 📌 **Training 과정**
1. **각 Epoch마다 YOLO는 이미지를 S x S 크기의 겹치지 않는 Grid로 나눈다.**
2. **각 Grid Cell이 포함하는 객체의 중심을 기준으로 탐지 확률 P(Object)를 계산하고, 가중치 Parameter를 기반으로 Bounding Box를 생성한다.**  
   - ⚠️ **Anchor Box 개념 도입(YOLOv2 이후)을 함께 공부할 것!**
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
  - P(Object): 해당 Grid Cell 안에 객체가 존재할 확률
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


## 🎯 **YOLOv1 핵심 요약**
- **이미지를 S x S 크기의 Grid로 나누고, 각 Grid Cell에서 Bounding Box를 예측하는 방식**
- **Bounding Box의 Confidence Score를 계산하여, 특정 Threshold 이상인 박스만 유지**
- **Loss를 통해 Bounding Box의 위치, Confidence Score, 클래스 예측을 최적화**
- **Validation 단계에서 NMS를 적용하여 최종 Bounding Box를 선택**
- **YOLOv2 이후부터는 Anchor Box를 도입하여 다양한 크기의 객체 탐지가 가능하도록 개선됨**

---

## ✅ **10년 뒤에도 쉽게 이해할 수 있도록!**
📌 **YOLOv1은 Bounding Box를 직접 예측하는 방식이었지만, YOLOv2 이후부터는 Anchor Box를 도입하여 더욱 정교한 탐지가 가능해졌다.**  
📌 **YOLO는 한 번의 Forward Pass로 전체 이미지에서 객체를 탐지할 수 있도록 설계된 실시간 Object Detection 모델이다.**  
📌 **이 문서를 다시 읽을 때는 YOLO 최신 버전과 비교하면서 차이점을 분석해볼 것!**
