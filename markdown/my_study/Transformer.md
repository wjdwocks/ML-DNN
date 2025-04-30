
# Transformer 공부 Attention is All You Need 논문
## 1. Transformer가 등장한 이유
### 기존 RNN, LSTM의 한계점
1. **긴 문장을 다룰 때 정보 손실 문제**
   - RNN, LSTM은 **입력을 순차적으로 처리**하므로 **긴 문장에서 앞쪽 정보가 뒤쪽에 잘 전달되지 않음**  
   - (Long-Term Dependency 문제 발생)
   
2. **병렬 연산 불가능 → 학습 속도 느림**
   - RNN/LSTM은 순차적으로 학습되므로, **GPU 병렬 연산이 어려움 → 학습 속도가 느림**
   
3. **Gradient Vanishing 문제**
   - 깊은 네트워크에서 **초반 입력의 Gradient가 사라지는 문제 발생 → 학습 어려움**

---

### Transformer는 어떻게 해결했나?
1. **Attention Mechanism (Self-Attention) 활용**
   - 입력 Sequence 내의 모든 단어가 각각 **다른 단어와 얼마나 연관이 있는지 동시에 계산**  
   - → **멀리 떨어진 단어들 간의 관계도 효과적으로 학습 가능**
   
2. **순차 처리(X), 병렬 연산(O) 가능**
   - 모든 단어를 **동시에 처리 가능 → GPU 가속 최적화**
   
3. **Residual Connection 사용 → Gradient Vanishing 방지**
   - 각 Layer에서 **원본 입력을 더해줌** → 정보 손실 방지

---

## 2. Transformer의 Encoder 주요 구성 요소.
1. **Token Embedding + Positional Encoding**
   - 단어(Token)을 고차원 벡터로 변환한 후, 순서 정보를 나타내는 Positional Encoding을 더해 Transformer가 순서를 인식할 수 있게 함.
2. **Multi-Head Self-Attention**
   - 각 Token이 문장 내 모든 Token과의 관계(유사도)를 계산하여 문맥을 반영한 벡터로 재구성함.
   - 여러 개의 Head를 사용해 다양한 관점(의미적, 문법적, 위치적 등)에서 Attention을 수행.
   - Head는 각 단어의 어떤 면에 집중할 것인지를 의미함.
   - 여기까지가 입력 Token의 vector표현을 위한 전처리라고 생각하면 됨.
3. **Feed Forward Network (FFN)** 
   - 각 Token에 독립적으로 적용되는 2 layer의 비선형 신경망
   - Attention을 통해 정제된 벡터 표현을 더 고차원적으로 가공하며, 모델이 복잡한 의미 표현을 학습할 수 있도록 **표현력**을 증가시킴.
   - 여기서 말하는 표현력이란, 각 Token의 Embedding Vector를 미세하게 조정하여 최적의 값을 가지도록 한다는 의미.
4. **Residual Connection + Layer Normalization** 
   - 각 블록 출력에 입력을 더해주는 Skip Connection을 통해 정보 흐름을 유지하고, Layer Normalization을 통해 학습 안정성과 수렴 속도 향상.
5. **결과 : Contextualized Representation**
   - 입력 문장 내 각 Token은 이제 문맥 정보를 반영한 벡터로 변환되어, Decoder나 후속 작업에서 사용할 수 있는 **문맥 Embedding**으로서 완성됨.
6. **Encoder의 주요 목표**
   - 입력 문장의 각 Token을 문맥 정보가 반영된 벡터로 변환하는 역할을 함.
   - 그에 따라서 Encoder의 출력은 input으로 들어온 Sequence(token들)를 문맥 정보가 포함된 Vector 표현으로 바꾼 Vector들이 된다.
   - 이 출력 Vector들은 Decoder에서 Cross-Attention의 Key/Value로 사용이 된다.

---

## 3. Transformer의 Encoder 동작 과정 (입력 → Context 벡터 생성)

### 예제: 입력 문장
> **"I am a teacher" (영어) → "나는 선생님이다" (한글)로 번역한다고 가정**

### Step 1: Input Embedding + Positional Encoding
- 입력 Sequence(문장)을 Tokenization을 함. → 'I', 'am', 'a', 'teacher'
- 각 단어(Token)를 **Embedding 벡터(고차원 표현)로 변환**
- Transformer는 단어의 순서를 알 수 없기 때문에 **Positional Encoding 추가**
- 최종 입력 벡터 = `Embedding Vector + Positional Encoding`

---

### Step 2: Multi-Head Self-Attention (자기 자신에게 Attention)
> **"I am a teacher" 문장에서 단어들 간의 연관성을 학습**

 **Query, Key, Value 행렬 생성**
- Self-Attention을 수행하여 **입력 Sequence 문장 내 각 단어가 문장 내 다른 단어와 얼마나 관련 있는지 계산**

 **Scaled Dot-Product Attention 수행**
- Attention Score 계산: Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
- Softmax를 적용하여 **모든 단어의 중요도를 확률적으로 정규화하고, 각 중요도에 따라 가중치를 부여**
- **모든 단어의 Value를 가중합하여 새로운 벡터를 생성**

 **Multi-Head Attention (h개로 분할하여 다른 의미 학습)**
- 각 Head는 **서로 다른 문맥 정보를 학습** (CNN의 필터처럼)
- 여러 개의 Attention 결과를 Concat하여 최종 Attention 벡터 생성

---

###  Step 3: Feed Forward Network (FFN)
> **Self-Attention을 거친 벡터를 더욱 정제된 표현으로 변환**

 **비선형 변환을 추가하여 표현력을 증가**
- FFN(X) = max(0, X * W1 + b1) * W2 + b2
- CNN과 DNN의 활성화 함수처럼, FFN에서도 비선형성을 추가하여 모델의 표현력을 증가시킴
- 더 풍부한 특징을 학습할 수 있도록 변환

 **Residual Connection과 Layer Normalization을 수행하여 정보 손실 방지**

---
## 4. Decoder의 주요 구성
1. **Token Embedding + Positional Encoding**
   - Predict : [Start of Sentence 또는 이전 출력 Sequence]를 고차원 벡터로 변환하고, 위치 정보를 더함.
   - 학습 시 : 정답 전체 Sequence를 입력으로 사용한다.
2. **Masked Self-Attention**
   - 기존 Encoder와 동일한 Self-Attention 구조지만, 미래의 단어를 보지 못하도록 마스킹함.
   - Test(Predict) 시 현재까지 출력한 Sequence 내에서만 Self-Attention을 수행
   - ex) 학습 시 3번째 단어를 예측할 때에는 1~2번 단어까지만 Self-Attention을 수행. 
3. **Cross Attention**
   - Encoder에서 나온 입력 문장의 Context Vector를 K, V로, Decoder에서 나온 이번 단어 Vector를 Query로 입력 문장을 참조함.
   - Decoder가 입력 Sequence의 의미를 활용할 수 있도록 도와주는 핵심 모듈임.
   - 즉, Cross Attention을 안하면, Encoder가 있을 필요가 없음. (Encoder의 Attention 가중치와 Decoder의 Attention 가중치는 각각 따로 학습됨.)
4. **Feed Forward Network(FFN)**
   - Encoder와 동일하게 각 위치에 독립적으로 적용되는 비선형 MLP Layer
5. **Residual Connection + LayerNorm**
   - 각 레이어 블록마다 안정적 학습과 정보 손실 방지를 위해 포함.
6. **Decoder의 마지막 출력 단어 벡터는 Softmax Layer를 거쳐 단어 분포로 변환**
   - Auto-Regressive 하게 한 단어씩 생성되며, 생성된 단어는 다시 Decoder에 입력되어 다음 단어 예측에 사용된다.


## 5. Transformer의 Decoder 동작 과정 (Context 벡터 → 문장 생성)

### Decoder의 입력
- 훈련할 때는 **정답 문장의 앞부분**이 Decoder의 입력이 됨
- 예측할 때는 **Decoder가 이전 step에서 예측한 단어 or SOS를 입력으로 사용**
- 예) `"<SOS> 나는"` → `"나는 선생님"` → `"나는 선생님이다"`

---

### Step 1: Decoder Input Embedding + Positional Encoding
- Decoder도 입력을 **Embedding 벡터로 변환 후, Positional Encoding을 추가**
- `"<SOS> 나는"` → Embedding → Positional Encoding 추가

---

### Step 2: Masked Multi-Head Self-Attention
> **Decoder가 "나는"을 예측할 때, "선생님"을 보지 못하도록 Masking 적용(Training 과정에서)**

 **Masked Self-Attention이란?**
- 현재까지 생성된 단어까지만 고려하도록 Future Token을 가리는 Mask 적용
- Softmax 연산 시, 미래 단어는 **-∞ (무한대로 낮은 값)** 처리하여 고려되지 않도록 함

---

###  Step 3: Encoder-Decoder Attention
> **Encoder에서 생성한 Context 벡터를 참고하여, 다음 단어를 예측**

 **Decoder의 Query, Encoder의 Key, Value 사용**
- Query: Decoder에서 Self-Attention을 수행한 후 벡터
- Key & Value: Encoder의 최종 출력 (입력 문장의 Context 벡터) # **Encoder의 Value를 사용한다는것은 꼭 기억해야 함.(Value는 Encoder의 정보임)**
- Decoder의 Query와 Encoder의 Key를 Attention 하여 나온 결과와 Encoder의 Value를 가중합한 결과를 FFN에 통과시킨 뒤 가장 가까운 단어를 찾으면 그것이 Predict 단어가 된다.

 **Encoder에서 얻은 Context 정보를 활용하여 더욱 정확한 단어 예측**
- `"나는"`이 등장했을 때, `"선생님"`이 나올 확률이 높아지도록 조정됨

---

###  Step 4: Feed Forward Network (FFN)
> **Attention을 수행한 벡터를 더욱 정제된 표현으로 변환**

 **Encoder와 동일한 FFN 적용**
- ReLU 활성화 함수 + 두 개의 Linear Layer 사용
- 최종적으로 문맥 정보를 더욱 잘 반영한 벡터를 생성

---

###  Step 5: Softmax를 통해 단어 예측
> **Decoder의 최종 출력을 Softmax에 통과시켜 가장 적절한 단어를 예측**

 **Softmax를 통해 확률 분포 계산**
- P(선생님 | 나는) = exp(z_선생님) / sum(exp(z_i))
- 가장 확률이 높은 단어 `"선생님"`을 선택

 **이전까지 생성된 단어를 입력으로 넣고 반복하여 최종 문장 생성**
- `"나는 선생님"` → `"나는 선생님이다"`

---

# Transformer 전체 과정 요약
1. **Encoder**
 - 입력 문장을 **위치 정보를 포함한 벡터로 변환**
 - Multihead-Self-Attention을 수행하여 문장 내 단어 간 관계 학습(이 때 Attention 가중치도 함께 학습이 진행됨.)
 - FFN을 통해 더욱 정제된 벡터로 변환

2. **Decoder**
 - 이전까지 생성된 Sequence or SOS를 입력으로 사용
 - 입력을 위치 정보를 포함한 벡터로 변환
 - Masked Self-Attention을 수행하여 현재까지 만들어진 Sequence 내 단어 간 관계를 학습.
 - Cross Attention을 통해 Encoder에서 생성한 벡터와의 관계를 학습.
 - FFN을 통해 최적의 벡터 생성 후 Softmax를 통해 단어 예측