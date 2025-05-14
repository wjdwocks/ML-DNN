# 여러 다중 Teacher 지식 증류 기법들 정리

## 1. Self-Paced Knowledge Distillation (자기-페이스 지식 증류, LFME, 2020)

- **제안 배경**  
  데이터 불균형(Long-tail 분포) 상황에서 모든 Teacher의 예측을 동일하게 사용하는 경우, Student는 Teacher들의 부정확하거나 불확실한 soft label까지 무비판적으로 학습하게 됨.  
  특히 Student가 학습 초기일 때는 복잡한 soft label을 따라갈 수 있는 capacity가 부족해 오히려 학습에 방해가 되거나 과적합으로 이어질 수 있음.  
  → 이를 방지하기 위해, Student가 이해하기 쉬운 Teacher와 샘플부터 점진적으로 학습하는  
  **Self-Paced curriculum 구조의 KD** 방식을 제안함.

- **작동 과정**  
  LFME(Learning From Multiple Experts)는 두 단계의 Self-Paced 전략을 사용하여 Teacher 및 샘플을 선택함:

  1. **Self-Paced Expert Selection**:  
     - Student가 현재 가장 잘 따라가고 있는 Teacher를 선택 (KL Divergence 기준)  
     - Epoch이 진행됨에 따라 KL 임계값을 완화시켜 더 많은 Teacher를 포함  
     - Teacher 간 KD Loss 가중치(α_k)는 고정되거나 softmax 정규화로 유동적으로 조정 가능
     - 만약 내가 사용한다면, 임게값을 넘는다면 base에서 얻은 alpha값을 그대로 사용하는것도 나쁘지 않을듯 하다.

  2. **Curriculum Instance Selection**:  
     - 선택된 Teacher의 soft label과 Student의 예측 간 KL Divergence가  
       낮은 샘플부터 선택하여 학습  
     - 점진적으로 어려운 샘플(즉, 예측 차이가 큰 샘플)도 포함되도록 임계값 증가

  - 실질적으로는 각 epoch마다 다음을 수행함:
    - 각 Teacher에 대해 Student와의 KL Divergence 계산
    - 현재 기준 이하인 Teacher만 선택하여 KD Loss 계산
    - 이후 학습이 진행될수록 더 많은 Teacher를 포함

- **장점**  
  - Student가 감당 가능한 수준의 지식부터 받아들일 수 있어 **학습 안정성 향상**
  - Teacher 예측이 부정확한 경우에도 이를 필터링할 수 있어 **노이즈에 강함**
  - Curriculum 구조 덕분에 **특히 데이터 불균형 상황에서 효과적**
  - Student가 복잡한 지식을 **점진적으로 흡수**할 수 있도록 설계됨

- **멀티모달 다중 Teacher 사용 가능 여부**  
  가능 (PI, GAF, Sig의 Teacher 난이도를 기반으로 순차 학습 가능)
  또한, 각 Teacher가 특정 클래스에 대한 Confidence가 높을 수록 성능이 좋다.

---

## 2. Confidence-Aware Multi-Teacher KD (CA-MKD, 2022)

- **제안 배경**  
  다중 Teacher 모델을 단순 평균하여 지식 증류를 수행할 경우,  
  각 Teacher의 예측 품질이 고르지 않으면 오히려 Student 모델에 **혼란을 주거나 성능을 저하시킬 수 있음**.  
  예를 들어, 어떤 샘플은 Teacher A가 잘 예측하고, 다른 샘플은 Teacher B가 더 나은데  
  모든 Teacher를 동일한 가중치로 반영하면 부정확한 지식이 주입될 수 있음.  
  → 이를 해결하기 위해 **샘플 단위로 Teacher의 신뢰도(confidence)를 평가하고**,  
  그에 따라 KD loss에 **가중치를 다르게 적용**하는 구조를 제안.

- **방식 요약**  
  - **Teacher confidence 측정**:  
    각 Teacher의 예측과 정답 간 Cross-Entropy를 기반으로 confidence score 계산  
    → 낮은 CE → 높은 confidence
  - **샘플 단위 confidence-weighted KD**:  
    각 샘플에 대해 Teacher별 confidence를 softmax 정규화 후,  
    **해당 샘플의 KD loss는 confidence-weighted sum으로 계산**
  - **Feature-level KD**:  
    최종 logit만 사용하는 것이 아니라, Teacher와 Student의 **중간 feature map** 사이의 차이도  
    L2 distance 등을 사용하여 추가적인 KD loss로 활용  
    → 모델 내부 표현까지 전달함으로써 학습 안정성 향상

- **구현 고려사항**  
  - 학습은 **batch 단위**로 진행되지만,  
    KD loss는 반드시 **샘플 단위로 confidence-weighted aggregation** 후 평균해야 함  
    (그렇지 않으면 CA-MKD의 설계 철학이 무력화됨)
  - Feature distillation은 **특정 layer의 feature map**을 정렬하는 방식으로 진행되며,  
    일반적으로 L2 loss 또는 cosine similarity 기반

- **장점**  
  - 신뢰도 낮은 Teacher의 영향을 억제하여 **오류 전파 감소**
  - **샘플마다 가장 적합한 Teacher의 지식만 반영**하여 학습 효율 향상
  - 중간 표현까지 모방함으로써 **내부 표현 정렬 및 일반화 성능 강화**
  - 다양한 Teacher 조합 (다른 도메인/모달리티) 상황에도 유연하게 대응 가능

- **멀티모달 다중 Teacher 사용 가능 여부**  
  매우 적합 (각 modality의 Teacher 신뢰도 차이를 반영 가능)

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
