# GAN 공부
## VAE 랑 GAN 비교
- VAE
    * 입력 x를 잘 복원하면서, Latent Space의 확률 분포를 구조적을 잘 구성함.
    * 어디에서 z를 샘플링하더라도 그럴듯한 데이터를 생성할 수 있게 만드는 것이 목표임.
- GAN (Generative Adversarial Network)
    * 임의의 noise vector z ~ N(0, 1)에서 시작해서, 판별기를 속일 정도로 진짜처럼 보이는 이미지를 생성하는 것이 목표임.

## GAN의 구조
- GAN의 핵심 아이디어
    * 두 개의 네트워크가 서로 경쟁하는 구조를 가진다.
    * Generator(G) : 무작위 벡터 z ~ N(0, 1)로부터 진짜처럼 보이는 이미지를 생성.
    * Discriminator(D) : 이미지가 진짜인지, Generator가 만든 가짜인지를 구별함.
    → Generator는 Discriminator를 속이는 방향으로 학습하고,
    → Discriminator는 속지 않는 방향으로 학습한다.

- 네트워크 구성
