### 전처리된 Text 입력값
- 각 Action Labels에 대해서 아레의 것들이 5개씩 저장됨. (merged_text)
    - Arm: Wrist swings slightly forward with low amplitude. Leg: Heel contacts ground initiating slow step.
    - Arm: Forearm passes alongside torso during forward swing. Leg: Body weight transfers to front foot.
    - Arm: Wrist reaches furthest forward point before reversing. Leg: Rear leg begins lift-off.
    - Arm: Wrist moves backward smoothly behind hip. Leg: Foot swings slowly forward through the air.
    - Arm: Forearm returns to neutral beside torso. Leg: Foot approaches ground preparing for next contact.

- 저장된 processed_data['Treadmill 1mph (0% grade)']로 접근하면, 아레와 같이 나옴.
    ```python
    ['Arm: Wrist swings slightly forward with low amplitude. Leg: Heel contacts ground initiating slow step.', 'Arm: Forearm passes alongside torso during forward swing. Leg: Body weight transfers to front foot.', 'Arm: Wrist reaches furthest forward point before reversing. Leg: Rear leg begins lift-off.', 'Arm: Wrist moves backward smoothly behind hip. Leg: Foot swings slowly forward through the air.', 'Arm: Forearm returns to neutral beside torso. Leg: Foot approaches ground preparing for next contact.']
    ```

- 기본 BERT와, BERT 기반이라는 sentence-transformers/all-MiniLM-L6-v2 이 두개로 Embedding을 만들어봄.
    - BERT는 (-1, 768)의 크기로 embedding이 생성되고
    - MiniLM-L6-v2는 (-1, 384)의 크기로 embedding이 됨.
    - 그냥 둘 다 npy, pt 두개씩 저장함.

### 직접 잘 되었는지 확인해보자.
- 직접 npy로 불러와서 "Treadmill 1mph (0% grade)" 이 Action에 대해서 Shape과 실제 값을 찍어보자.
    - Bert로 저장된 "Treadmill 1mph (0% grade)"의 Shape : (5, 768) → 예상한 것과 동일
    - MiniLM으로 저장된 "Treadmill 1mph (0% grade)"의 Shape : (5, 384) → 예상한 것과 동일
    - Bert로 저장된 "Treadmill 1mph (0% grade)"의 첫 Sub-Shape의 첫 5개 벡터값 : [-0.25430912 -0.23059809  0.3496979  -0.22666736 -0.10425091]
    - MiniLM으로 저장된 "Treadmill 1mph (0% grade)"의 첫 Sub-Shape의 첫 5개 벡터값 : [-0.03090179 -0.04691923  0.04170857 -0.02017843 -0.09610803]
- 값 자체는 MiniLM이 더 작은 느낌? 둘 다 해보지 뭐
- LLM Text Encoder를 이용해서 Embedding Vector로 만드는 것 까지는 완.

### 이 벡터들을 t-sne, umap, pca를 통해서 2차원 및 3차원으로 시각화해보자.
- sub-actions를 Arm에 대한 묘사만 있을 때, Leg에 대한 묘사만 있을 때, Arm + Leg가 한 문장으로 통으로 들어갈 때로 나누어서 봐봄.
    - t-sne(2차원)
        - <img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/11월/25.11.21/bert_cluster_arm.png" alt="results" width="700">
        - <img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/11월/25.11.21/bert_cluster_leg.png" alt="results" width="700">
        - <img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/11월/25.11.21/bert_cluster_armleg.png" alt="results" width="700">
    - umap(3차원), n_neighbors = 70(전체 구조 유지를 위함)
        - <img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/11월/25.11.21/bert_umap_arm.png" alt="results" width="700">
        - <img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/11월/25.11.21/bert_umap_leg.png" alt="results" width="700">
        - <img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/11월/25.11.21/bert_umap_armleg.png" alt="results" width="700">
    - pca(3차원)
        - <img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/11월/25.11.21/bert_pca_arm.png" alt="results" width="700">
        - <img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/11월/25.11.21/bert_pca_leg.png" alt="results" width="700">
        - <img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/11월/25.11.21/bert_pca_armleg.png" alt="results" width="700">

- 우선, 같은 label이라고 같이 있으면 안좋을 것임. (비슷한거를 제거하고, 몇개만 남길텐데, 그러면 어려워질 수도.)
- 다른 Label에서 나왔더라도, 비슷한 Label이었으면, 특정 Sub-Action끼리는 비슷한게 당연함.
- Drive Car 같은 경우는 Leg에 대한 움직임이 거의 없을 것이기 때문에, Leg만 봤을 때 비슷하게 있어야 맞을듯
- 근데, 어쨋든 우리는 많은 정보가 있으면 좋으니까 arm+leg를 한거를 기준으로 보는게 맞을듯 하긴 합니다?


### 이 벡터들을 이용해서 간단한 Reconstruction을 수행하는 네트워크를 이용해서 각 Sub-Action만으로 원본 Label을 잘 유추할 수 있을지 학습해보기.
1. 우선, 아주 작은 간단한 MLP를 하나 만듦.
    - 모델은 아레와 같이 간단한 Linear Layer 2개를 이어서 만들었다.
```python
class ProjectionModel(nn.Module):
    """
    Concat-based Projection:
    Input: concat(5 sub-actions) = 3840 dim
    Output: 768 dim action embedding
    """
    def __init__(self, sub_dim=768, num_sub=5, hidden_dim=256): # Hidden_dim = 64, 128, 256, 512, 1024, 2048...
        super().__init__()
        input_dim = sub_dim * num_sub       # 768*5 = 3840

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sub_dim)
        )

    def forward(self, x):
        return self.mlp(x)
```
2. Sub-Actions의 Embedding 정보들을 합친 후 MLP를 통과시켜 원래 Label의 Embedding을 맞추도록 설계
    - Action Label또한 같은 LM에 넣어서 Embedding을 추출하고, 이것을 Label로 Reconstruction Loss를 설계함.
    - Sub-Actions 5개의 Embedding을 concat해서 입력으로 넣고, 768차원의 vector를 예측하도록.

3. Train / Test Set을 나눔.
    - 원래로는 Training만을 생각했는데, 당연하게도, overfitting에 의해서 다 맞출 수 밖에 없다.
    - 14class짜리 5개씩 sub-action 생성해도 sample이 70개밖에 안되기 때문에
    - 그래서, Train과 Test set을 나눠보았다. (그냥 안될거라고 생각하고 해봤다.)
    - 근데, 생각보다 어느정도 Test set에 대해서 Embedding 간의 cosine similarity가 높게 나왔음. (bert-base에서는 어느정도 높게 나왔는데, minilm에서는 좀 낮음.)
    - <img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/11월/25.11.21/recon_bert.png" alt="results" width="700">
    - <img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/11월/25.11.21/recon_minilm.png" alt="results" width="700">


### 추출된 70개의 Sub-Action간에 Gram-Matrix를 통해서 서로 간의 Similarity를 정량적으로도 보자.
- 이거 보고, Sub-Action 출력하는거 다시 Prompt 점검 해봐야함. (다른 것들도 출력해보고, Sub-Action을 직접 계속 읽어보며 논리적으로도 확인.)
- 위 두가지를 계속 해보면서 지금까지의 과정 계속 추가.
- VQShape에 GENEActiv로 돌려보기.
- VQShape에 이 과정을 추가해서 Framework 넣어보기. (가능하다면..)

### Sub-Action을 10개씩 추출하게 바꿔보자.
- Shape Codebook이 64개라고 생각하면, Action도 64개로 하고싶은데, 그럴라면 10개씩 140개 만들면 좋을 듯.

### Sub-Action 추출할 때 "Arm : , Leg : " 이런식으로 했었는데, 안나누고 그냥 해보기.
- GPT가 Arm, Leg 를 안나누고 추출해달라고 하면, 어디에 집중해서 추출해줄지 궁금하긴 함.
- 추출해보고, Recon, Clustering 다 해서 시각화까지.
- 다 했는데, 그림이 너무 많아져서 주간보고때 다 보여드리겠습니다.


### 기존 VQShape에 GENE_Activ로 돌려보기.
- 일단 VQShape에 GENE_Activ 적용해서 돌려봐야 그 다음이 됨.
- VQShape 논문에서, Multi-Variate TS를 Channel로 나누어서 여러 개의 Univariate TS로 나누었다는 것으로 이해함.
    - 그래서, GENEActiv 데이터셋을 (1, 500, 3)으로 되어있는 것을 우선 (3, 500)으로 나누고, 그에 대응하는 Label을 복사해서 넣어줌.
    - 즉, 샘플 하나당 (1, 500, 3) + Label 이었던 것을 (500, ) + Label 로 세개로 늘렸다.
    - <img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/11월/25.11.21/Multi-process.png" alt="results" width="700">
    - 그 다음으로, VQShape의 기존 코드들을 대부분 그대로 사용하기 위해서 논문에서 사용한 전처리를 그대로 적용함. 논문에서는 Time-Series를 512의 Length로 잘라서 사용했다는 취지로 Interpolate했다는 것 같은데, 나는 GENEActiv를 그대로 사용할 것이라서, 500window짜리를 linear interpolate로 512로 늘려줬다. UEA에서 그렇게 한 것 같길래...
    - <img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/11월/25.11.21/interpolate.png" alt="results" width="700">

### 위의 방식의 문제점 및 어떻게 할지
1. VQShape에서 Multi-Variate를 다룬 방식
    - 애초에 VQShape의 Pretrain에서는, Codebook을 만드는 것이 목표이기 때문에, 각 축마다 다른 shape이 나와도 큰 상관이 없었음. 그냥 그 각 Channel마다 어떤 Shape이 나왔고, 그것들을 잘 이용해서 Downstream Task에 사용. (forecasting, classification ...)
    - Pretrain에서는 Codebook을 잘 만드는 것에만 집중을 하고, 각 Channel마다 이런 Shape이 나오고, 그것들을 조합해보면 이렇다. 라고 할 수가 있음.

2. GENEActiv Classification에 적용하려고 하는데 바로는 당연히 안됨.
    - GENEActiv를 통해서 Classification을 한다고 생각하면, 이 채널별로 나뉘어진 것들이 같은 Label을 의미하게 해야 함.
    - 원래라면, 전체 시계열 재구성 loss를 빼고, 나는 Classification Loss를 넣으려고 했는데, 이렇게 할 수가 없다. why? : 각 x, y, z 축 별로 완전히 다른 모양인데, 같다고 해야하기 때문에.
    - 그리고, 각 축 하나의 표현만으로는 Action도 정확하게 예측할 수가 없지 당연히...

3. 어떻게 해야 할까?
    - (3, 512)로 DataLoader가 출력하게 함. (즉, univariate로 나눠서 interpolation까지는 하되, 다시 3채널을 합쳐서 하나의 샘플로)
    - LSA쪽의 과정을 통해서 Signal Token까지 만들어 졌다는 가정 하에, 각 채널별로 Attribute Decoder를 통과시켜서 Codebook관련, sub-shape Reconstruction 관련 연산을 별개로 수행함.
    - 그리고, 다시 channel-wise로 분할된 attributes들을 다시 합쳐서 Attribute Encoder 에 넣은 후 Classifier에 넣어서 End-to-End 학습을 하거나, x_reconstruction을 수행해서 pretrain만 함.
    - 거기에 Text Token은, LSA에 의해서 Text Token이 잘 만들어졌다는 가정 하에, Text 전용 Attribute Decoder에 각 Token들을 통과시켜서 나온 Text Embedding 들과 Text Codebook내의 Embedding들을 비교하여 가장 비슷한 것으로 대체한 후, 이 5개를 concat하여 Projection Layer에 통과시킨 후 원본 Label Sample과 Reconstruction을 수행한다.

4. DataLoader가 내보내는 형태
    - (Signal, Label, Sub-Actions, Label-Embedding)로 데이터 입력을 넣어줌.
    - Signal : (3, 512)로 interpolate하고, 다시 channel-wise로 합쳐진 형태
    - Label : 정수 하나 그대로.
    - Sub-Actions : 5개 ~ 10개 사이의 Text Embedding을 보내줌.
    - Label-Embedding : 해당 Label에 해당하는 Text의 Embedding

5. 그래서 어떻게 학습 할건데?
    - 