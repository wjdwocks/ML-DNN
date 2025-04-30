# Vision Transformer(ViT) 공부
## 1. Vision Transformer의 입력 처리
### 기존 Transformer와 다른 점.
- 기존 Transformer는 자연어 문장을 Token(단어) 단위로 나눈 뒤, Token Embedding + Positional Encoding을 거쳐 입력으로 사용함.
- 입력이 시퀀스(Sequence) 형태로 구성되기 때문에, Attention 연산이 Token 간 관계(문맥)를 자연스럽게 학습할 수 있음.
- 하지만 이미지는 시퀀스가 아니기 때문에, Transformer에 바로 입력할 수 없음.
  → 따라서 이미지를 시퀀스로 바꾸는 전처리 과정이 필요함.

---

### ViT에서의 입력 처리

1. **이미지를 Patch로 분할**
   - 입력 이미지 (예: 224×224 RGB)를 고정된 크기(예: 16×16)의 Patch로 분할
   - 하나의 이미지는 \( (224 / 16)^2 = 196 \)개의 Patch로 나뉘게 됨

2. **각 Patch를 Flatten**
   - 각 Patch는 16×16×3 = 768 차원의 1D 벡터로 펼쳐짐

3. **Linear Projection (Patch Embedding)**
   - Flatten된 Patch를 고정 차원의 Embedding Vector로 변환
   - 각 Patch는 Token처럼 취급되어 Sequence 형태로 재구성됨

4. **[CLS] Token 추가 (선택 사항)**
   - 전체 이미지를 대표하는 벡터로 활용하기 위해 가장 앞에 Special Token인 [CLS]를 추가할 수 있음
   - CLS는 전체 Sequence를 대표할 수 있는 벡터 하나를 만들기 위한 특수 토큰이다. (BERT 처음 등장했다고 함.)
   - 왜 필요함? → Transformer는 기본적으로 입력 Sequence의 각 Token마다 하나의 출력 벡터를 생성함. → 근데 어떤 Task는 문장 전체에 대한 판단이 필요 (ex. 리뷰 : 긍정/부정, 이미지 분류 : 고양이/개)
   - 그래서 각 Sequence 전체를 대표하는 CLS라는 가상의 Token(Vector)을 추가해서 Transformer가 이 Token의 출력만 보고 분류를 수행하도록 학습시킴.
   - Classification Task에서 중요하게 쓰인다고 함....

5. **Positional Encoding 추가**
   - 순서 정보가 없는 Transformer에 위치 정보를 주기 위해, Patch Embedding에 Positional Encoding을 더함

---

### 처리 후 과정
- 위 과정(Patch 분할 → Flatten → Embedding → Positional Encoding → [CLS] Token 추가)을 통해 만들어진 입력은  
  기존 NLP Transformer에서 사용하던 Token Sequence와 구조적으로 동일한 형태가 된다.  
  즉, 각 Patch는 하나의 Token처럼 취급되며, 순서 정보도 함께 포함된 상태다.

- 이 시점부터는 기존 Transformer Encoder와 완전히 동일한 방식으로 동작한다:

  - **Multi-Head Self-Attention**  
    각 Patch Token은 자신을 포함한 모든 다른 Patch Token들과의 관계(유사도)를 계산한다.  
    이 과정을 통해 각 패치가 이미지 내 다른 패치들과 어떻게 연관되어 있는지를 학습하며,  
    이를 바탕으로 문맥(Context)이 반영된 새로운 표현(벡터)로 업데이트된다.  
    여러 개의 Head가 병렬적으로 서로 다른 관점(색상, 경계, 질감 등)을 학습하며 이를 종합함.

  - **Feed Forward Network (FFN)**  
    Attention으로 정제된 각 Token 벡터에 대해 독립적으로 비선형 변환을 수행한다.  
    이 과정을 통해 더욱 복잡한 표현력(Representation Power)이 추가된다.

  - **Residual Connection + Layer Normalization**  
    Attention과 FFN 이후에는 입력과 출력을 더해주는 Skip Connection(잔차 연결)과  
    Layer Normalization이 적용된다.  
    이를 통해 학습의 안정성과 정보 흐름의 보존을 동시에 달성한다.

- 이러한 Encoder Block(= Attention + FFN + Norm)의 구성은 여러 층으로 반복되며,  
  각 Token의 표현이 점점 더 풍부하고 정교해진다.

- **최종 출력 단계에서는 다음과 같이 활용된다:**

  - 분류(Classification) 작업의 경우:  
    가장 앞에 넣었던 **[CLS] Token의 출력 벡터**만 추출하여,  
    해당 벡터를 **Linear Layer + Softmax**에 연결하여 **이미지 전체의 클래스(label)를 예측**한다.  
    ([CLS]는 이미지 전체를 요약하는 벡터로 학습되었기 때문)

  - 다른 작업 (예: Object Detection, Image Segmentation 등)의 경우:  
    전체 Token 시퀀스(= Patch별 출력 벡터들)를 활용하여,  
    각 위치마다의 객체 존재 여부, 경계 박스, 또는 마스크 등을 예측하는 구조로 확장 가능하다.