# GAN 공부
## VAE 랑 GAN 비교
- VAE
    * 입력 x를 잘 복원하면서, Latent Space의 확률 분포를 구조적을 잘 구성함.
    * 어디에서 z를 샘플링하더라도 그럴듯한 데이터를 생성할 수 있게 만드는 것이 목표임.
- GAN (Generative Adversarial Network)
    * 임의의 noise vector z ~ N(0, 1)에서 시작해서, 판별기를 속일 정도로 진짜처럼 보이는 이미지를 생성하는 것이 목표임.
    * GAN은 애초에 '생성형 모델' 이 아니라, '생성형 모델을 학습하는 방식'인 방법론 이다.

## GAN의 구조
- GAN의 핵심 아이디어
    * 두 개의 네트워크가 서로 경쟁하는 구조를 가진다.
    * Generator(G) : 무작위 벡터 z ~ N(0, 1)로부터 진짜처럼 보이는 이미지를 생성. (어떤 생성형 모델도 가능)
    * Discriminator(D) : 이미지가 진짜인지, Generator가 만든 가짜인지를 구별함. (CNN 분류기의 형태를 띔)
    → Generator는 Discriminator를 속이는 방향으로 학습하고,
    → Discriminator는 속지 않는 방향으로 학습한다.

- 네트워크 구성
    1. Generator
        * 입력 : z ~ N(0, 1)인 정규분포에서 샘플링.
        * 출력 : 이미지 x'
        * 보통 Fully Connected + Transposed Conv Layer로 이루어져 있다.
    2. Discriminator
        * 입력 : 이미지 x'
        * 출력 : 확률 (이미지가 1에 가까우면 Real, 0에 가까우면 Fake)
        * 역할 : 입력이 진짜인지, 가짜인지 판별함.
        * 보통 CNN기반의 분류기처럼 동작함.

## GAN의 학습 과정
- 입력
    * Generator는 noise vector z (예: 정규분포에서 샘플링된 벡터)를 입력으로 받아 fake 이미지를 생성한다.
    * Discriminator는 다음 두 가지 이미지를 입력으로 받는다:
        - Generator가 만든 fake 이미지
        - 실제 데이터셋에서 가져온 real 이미지

- 판별
    * Discriminator는 입력된 이미지가 real인지 fake인지 구별한다.
    * 이 판단 결과를 기반으로 Binary Cross Entropy Loss를 계산한다.

- 학습
    * Discriminator는 자신이 real/fake를 더 잘 맞추도록 파라미터를 업데이트한다.
    * Generator는 Discriminator가 자신이 만든 fake 이미지를 real이라고 착각하게 만들도록 학습된다.  
    * 즉, Discriminator의 출력이 1(진짜)에 가까워지도록 자신의 파라미터를 업데이트한다.
    * Generator는 Discriminator의 출력을 사용해서 간접적으로 역전파를 받음. → KD에서 Teacher의 logits을 받아서 Student의 Parameter를 업데이트하듯이 이것도 비슷한듯.

