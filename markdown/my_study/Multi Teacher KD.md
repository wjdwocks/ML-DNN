# 여러 다중 Teacher 지식 증류 기법들 정리

## 1. Self-Paced Knowledge Distillation (자기-페이스 지식 증류, LFME, 2020)

- **제안 배경**  
  데이터 불균형(Long-tail 분포) 상황에서 모든 Teacher의 예측을 동일하게 사용하는 경우,  
  Student는 정확하지 않은 soft label까지 무비판적으로 학습하게 되어 오히려 성능이 떨어질 수 있음.  
  특히 학습 초반에는 Student가 복잡한 Teacher 출력을 수용할 능력이 부족하기 때문에,  
  Teacher의 지식을 점진적으로 받아들이는 **커리큘럼 기반 지식 증류** 방식이 필요함.

- **방식 요약**

  1. **Teacher 구성**
     - 전체 데이터셋을 클래스 분포에 따라 여러 subset으로 나누고  
       각 subset을 기반으로 별도의 Teacher 모델을 학습시킴
     - 각 Teacher는 자신이 학습한 subset(class 그룹)에 대해 더 잘 예측하는 경향이 있음

  2. **Teacher 선택 및 가중치 조절**
     - Student는 학습 중, batch에 포함된 샘플이 어떤 subset 출신인지 확인하고  
       **해당 subset을 담당한 Teacher의 예측만 사용**
     - Teacher의 성능(정확도)과 Student의 현재 성능을 비교하여  
       **가중치(weight)**를 계산해 Teacher별 영향을 동적으로 조절  
     - → **정확한 Teacher는 더 많이 반영**, 성능이 비슷해지면 영향 줄임  
     → *(이 부분은 성능 기반 가중치 곡선 그림으로 대체 가능)*

  3. **샘플 선택 커리큘럼**
     - 각 subset 내부에서 샘플 난이도(confidence 기준)를 정렬하여  
       **쉬운 샘플부터 점진적으로 어려운 샘플로 학습 확장**
     - 학습 초기에 모든 subset에서 일부 샘플만 선택 → 이후 epoch 진행에 따라 포함 비율 증가  
     - → *(이 부분은 "점점 늘어나는 샘플 분포" 커리큘럼 타임라인 그림으로 대체 가능)*

- **장점**
  - Student가 **감당 가능한 Teacher와 샘플부터 학습**하기 때문에 과적합과 과부하 방지
  - **정확한 Teacher의 정보만 활용**하여 부정확한 지도 신호의 영향을 억제
  - Long-tail 상황에서도 **소수 클래스 샘플이 커리큘럼에 포함될 수 있도록 설계**
  - **Teacher 수가 많아도 유연하게 통제 가능**한 구조


- **멀티모달 다중 Teacher 사용 가능 여부**  
  가능 (PI, GAF, Sig의 Teacher 난이도를 기반으로 순차 학습 가능)
  또한, 각 Teacher가 특정 클래스에 대한 Confidence가 높을 수록 성능이 좋다.

---

## 2. Confidence-Aware Multi-Teacher KD (CA-MKD, 2022)

- **제안 배경**  
  다중 Teacher 모델을 단순 평균 혹은 고정 가중치를 이용하여 지식 증류를 수행할 경우, 각 Teacher의 예측 품질이 고르지 않으면 오히려 Student 모델에 **혼란을 주거나 성능을 저하시킬 수 있음**.  
  예를 들어, 어떤 샘플은 Teacher A가 잘 예측하고, 다른 샘플은 Teacher B가 더 나은데 모든 Teacher를 동일한 가중치로 반영하면 부정확한 지식이 주입될 수 있음.  
  → 이를 해결하기 위해 **샘플 단위로 Teacher의 신뢰도(confidence)를 평가하고**, 그에 따라 KD loss에 **가중치를 다르게 적용**하는 구조를 제안.

- **방식 요약**  
  - **Teacher confidence 측정**: 
    각 Teacher의 예측과 정답 간 Cross-Entropy를 기반으로 confidence score 계산 → 낮은 CE → 높은 confidence
  - **샘플 단위 confidence-weighted KD**: 
    각 샘플에 대해 Teacher별 confidence를 softmax 정규화 후, **해당 샘플의 KD loss는 confidence-weighted sum으로 계산**
  - **Feature-level KD**:  
    최종 logit만 사용하는 것이 아니라, Teacher와 Student의 **중간 feature map** 사이의 차이도 MSE 등을 사용하여 추가적인 KD loss로 활용 → 모델 내부 표현까지 전달함으로써 학습 안정성 향상

- **구현 고려사항**  
  - 학습은 **batch 단위**로 진행되지만, KD loss는 반드시 **샘플 단위로 Confidence Score를 계산** 후 가중 평균해야 함.(그렇지 않으면 CA-MKD의 설계 철학이 무력화됨)
  - Feature distillation은 **특정 layer의 feature map**을 정렬하는 방식으로 진행되며, 일반적으로 L2 loss(MSE) 또는 cosine similarity 기반으로 손실이 계산됨.

- **장점**  
  - 신뢰도 낮은 Teacher의 영향을 억제하여 **오류 전파 감소**
  - **샘플마다 가장 적합한 Teacher의 지식만 반영**하여 학습 효율 향상
  - 중간 표현까지 모방함으로써 **내부 표현 정렬 및 일반화 성능 강화**
  - 다양한 Teacher 조합 (다른 도메인/모달리티) 상황에도 유연하게 대응 가능

- **멀티모달 다중 Teacher 사용 가능 여부**  
    - 매우 적합 (각 modality의 Teacher 신뢰도 차이를 반영 가능)
    - PAMAP2를 보면, PI가 잘 맞추는 sbj가 있고, GAF가 잘 맞추는 sbj가 있고, Sig가 잘 맞추는 sbj가 다 달라서 매우 적합할듯. (내꺼보다 높으면 안되는데;;)
    https://github.com/Rorozhl/CA-MKD
---

## 3. Adaptive Ensemble KD in Gradient Space (AE-KD, 2020)

- **제안 배경**  
  Teacher 간 손실을 단순 평균하면 서로 상충되는 gradient가 발생 가능  
  → gradient 관점에서 최적 가중치를 찾는 방식 제안

- **방식 요약**  
  - Teacher마다 생성하는 gradient를 분석  
  - Pareto 최적 기반으로 Teacher 손실 가중 조절

- **장점**  
  - Teacher 간 상충 최소화  
  - 이론적으로 강력한 최적화 구조  
  - Student가 균형 잡힌 방향으로 학습 가능

- **멀티모달 다중 Teacher 사용 가능 여부**  
  가능 (다른 입력 형식이라도 gradient 기준 조합 가능)

---

## 4. Meta-Learning 기반 다중 Teacher KD (MMKD, 2023)

- **제안 배경**  
  무조건 성능 좋은 Teacher가 학생에게 최적인 것은 아님  
  → 학생의 특성과 샘플에 따라 Teacher 조합을 다르게 해야 한다는 점에서 출발

- **방식 요약**  
  - 메타-가중치 네트워크를 학습하여, 샘플별로 최적의 Teacher 가중치 생성  
  - logits, feature similarity, uncertainty 등을 입력으로 활용

- **장점**  
  - 샘플별 최적 조합 가능  
  - 맞춤형 지식 전달  
  - 어려운 샘플에 더 강함

- **멀티모달 다중 Teacher 사용 가능 여부**  
  매우 적합 (PI, GAF, Sig 간 구조 차이를 학습 기반으로 흡수 가능)

---

## 5. Entropy-Based Multi-Teacher KD (DE-MKD, 2023)

- **제안 배경**  
  Teacher의 예측 분포가 얼마나 확신 있는지를 기준으로 가중치를 조절하자는 아이디어

- **방식 요약**  
  - Teacher의 예측 엔트로피(불확실성)를 계산  
  - 낮은 엔트로피(= 높은 확신)를 가진 Teacher 예측에 더 높은 가중치  
  - logits 뿐 아니라 feature-level에서도 적용 가능

- **장점**  
  - 계산 단순함  
  - 잘못된 예측 억제  
  - 평균보다 더 세밀한 조절 가능

- **멀티모달 다중 Teacher 사용 가능 여부**  
  가능 (각 modality Teacher가 가진 확신도 차이를 엔트로피로 반영 가능)

---

## ✅ 요약 비교표

| 방법명     | 핵심 전략                       | 장점                              | 멀티모달 적용 |
|------------|----------------------------------|-----------------------------------|----------------|
| SPKD       | 쉬운 Teacher부터 순차 학습       | Curriculum 학습, 안정성           | ✔ 가능         |
| CA-MKD     | 신뢰도 기반 가중 평균            | 잘못된 Teacher 억제, 안정성       | ✔ 매우 적합    |
| AE-KD      | Gradient 기반 Pareto 조합       | Teacher 간 상충 최소화            | ✔ 가능         |
| MMKD       | 메타러닝으로 샘플별 가중치 조정 | 맞춤형 Teacher 조합, 학습 적응력  | ✔ 매우 적합    |
| DE-MKD     | Entropy 기반 Teacher 선택       | 단순하면서도 효과적, 정교한 조절  | ✔ 가능         |
