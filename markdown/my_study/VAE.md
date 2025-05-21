# VAE (Variational Auto Encoder)

## VAE와 AE의 차이
- Auto Encoder(AE)
    * 목표 : 주어진 입력 x를 latent vector z로 압축하고, 다시 복원하는 것.
    * Latent Vector z는 고정된 Deterministic한 Vector값임.
    * 무언가를 생성하는 목적에는 부적합하다.
- Variational Auto Encoder(VAE)
    * 목표 : 입력을 잘 복원하는 동시에, 그럴듯한 새로운 샘플을 생성할 수 있는 구조적인 Latent Space를 학습하는 것.
    * 즉, 단순히 정보의 압축이 아니라, 분포 전체를 모델링하려는 목적을 가짐. 
        - 분포 전체를 모델링한다는 것은 어디에서 샘플링하더라도 그럴듯한 이미지를 생성할 수 있게 하고 싶다는 의미임.
    * 그래서 Encoder의 출력은 하나의 고정된 Vector가 아닌, 분포의 Parameter (μ, σ²)을 출력한다.
        - 이 때 학습의 핵심은 훈련 데이터로부터 만들어진 다양한 분포들을 최대한 정규분포 N(0, 1)과 최대한 비슷하게 만들도록 훈련된다.
        - 그래야 학습된 Latent Space(분포)에서 N(0, 1)로 샘플링했을 때 그럴듯하게 나올 것이기 때문에.

## VAE의 Latent Vector
- VAE의 Encoder에서는 입력 x에 대해서 두 개의 출력을 내보낸다.
    * μ(x) = 해당 입력에 대한 평균 벡터
    * log σ²(x) = 분산 벡터의 로그 (로그는 그냥 계산 안정성과 양수 보장을 위해서 사용한다고 함.)
    - 즉, Encoder는 분포의 파라미터를 추정하는 네트워크로 동작함. 
    - 그리고, 뒤에서는 항상 Encoder의 출력이 정규 분포라고 가정하고, 학습이 진행된다.
- 학습 시 Encoder는 입력 x에 대해 의미 있는 μ와, σ²을 출력하도록 학습된다.

## Latent Vector에서의 샘플링
- 우리는 이제, Latent Vector에 대한 특정 분포를 알고 있다.
- 그 분포에서 하나의 Latent Vector를 샘플링함.
    * 이 때 샘플링된 벡터 z = μ + σ ⋅ ϵ 로 얻어진다.
    * ϵ는 N(0, 1)에서 샘플링된 값이다.
- 이 z가 이미지 전체의 정보를 대표하는 압축된 표현이 된다.

## VAE의 Decoder
- VAE의 Decoder는 하나의 압축된 Vector z를 원본 이미지로 복원한다.
- 여기서 복원하는 것도 그냥 Auto Encoder와 같이 원본처럼 잘 복원하도록 파라미터가 학습된다.

## VAE의 loss 함수
- 복원 손실 : Decoder에서 출력된 이미지가 원본 입력과 얼마나 같은지를 판단함. 
    * 각 픽셀마다 MSE를 통해서 총합 할 수도 있고, 각 픽셀이 (0, 1)로 정규화되어있다면 각 픽셀을 binary CE를 통해 총합할 수도 있음.
- 정규화 손실 : Encoder에서 생성한 Latent Vector의 분포 파라미터가 얼마나 정규 분포를 따르는지를 판단하는 손실
    * KL-Divergence를 통해 정규분포에 얼마나 근접한지를 판단하여 손실에 추가함.

## VAE의 Reparameterization Trick
- 위에서 z = μ + σ ⋅ ϵ 로 얻어진다고 했음.
    * 그렇다면 왜 z = torch.normal(mu, sigma)로 실제로 샘플링을 하지 않는가?
    * z의 샘플링 자체를 stochasitc 연산으로 처리를 해버리면, backpropagation 중에 gradient가 계산되지 않는다.
    * 그래서, 정규분포에서 ϵ를 샘플링해서 기본 평균값에서 노이즈를 추가해주는 느낌으로 구현을 해서 Torch가 Gradient를 역전파할 수 있게(학습할 수 있게) 바꿔준다.
    