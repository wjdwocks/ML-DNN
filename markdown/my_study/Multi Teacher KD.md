# 여러 다중 Teacher 지식 증류 기법들 정리

## 1. Self-Paced Knowledge Distillation (자기-페이스 지식 증류, LFME, 2020)

- **제안 배경**  
  데이터 불균형(Long-tail 분포)에서 학생 모델이 전체 Teacher 지식을 한꺼번에 받으면 과부하가 생김.  
  → 쉬운 Teacher부터 순차적으로 지식을 학습하는 self-paced 방식 제안.

- **방식 요약**  
  - 각 Teacher를 전문가(Expert)로 보고, 쉬운 Teacher부터 학습  
  - Curriculum 기반 2단계 증류 (Teacher 선택 + 샘플 선택)

- **장점**  
  - 과부하 없이 안정적인 학습  
  - Long-tail 분포에 강함  
  - 기존 KD보다 더 자연스러운 학습 흐름

- **멀티모달 다중 Teacher 사용 가능 여부**  
  ✔ 가능 (PI, GAF, Sig의 Teacher 난이도를 기반으로 순차 학습 가능)

---

## 2. Confidence-Aware Multi-Teacher KD (CA-MKD, 2022)

- **제안 배경**  
  Teacher의 예측 품질이 고르지 않으면 단순 평균은 오히려 성능 저하  
  → Teacher 신뢰도를 반영한 가중 평균 구조 제안

- **방식 요약**  
  - 각 Teacher의 예측에 대해 샘플 단위로 신뢰도(confidence) 측정  
  - 높은 신뢰도를 가진 Teacher의 예측만 강조  
  - 중간 feature map도 함께 활용

- **장점**  
  - 잘못된 Teacher 영향 억제  
  - 학습 안정성 향상  
  - 다양한 Teacher 조합에 유연하게 대응

- **멀티모달 다중 Teacher 사용 가능 여부**  
  ✔ 매우 적합 (각 modality의 Teacher 신뢰도 차이를 반영 가능)

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
  ✔ 가능 (다른 입력 형식이라도 gradient 기준 조합 가능)

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
  ✔ 매우 적합 (PI, GAF, Sig 간 구조 차이를 학습 기반으로 흡수 가능)

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
  ✔ 가능 (각 modality Teacher가 가진 확신도 차이를 엔트로피로 반영 가능)

---

## ✅ 요약 비교표

| 방법명     | 핵심 전략                       | 장점                              | 멀티모달 적용 |
|------------|----------------------------------|-----------------------------------|----------------|
| SPKD       | 쉬운 Teacher부터 순차 학습       | Curriculum 학습, 안정성           | ✔ 가능         |
| CA-MKD     | 신뢰도 기반 가중 평균            | 잘못된 Teacher 억제, 안정성       | ✔ 매우 적합    |
| AE-KD      | Gradient 기반 Pareto 조합       | Teacher 간 상충 최소화            | ✔ 가능         |
| MMKD       | 메타러닝으로 샘플별 가중치 조정 | 맞춤형 Teacher 조합, 학습 적응력  | ✔ 매우 적합    |
| DE-MKD     | Entropy 기반 Teacher 선택       | 단순하면서도 효과적, 정교한 조절  | ✔ 가능         |
