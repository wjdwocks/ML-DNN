# 🧠 Diffusion Model + Text-to-Image + Video Generation 로드맵

---

## ✅ 1단계: Diffusion Model 기본기 확립

- [ ] DDPM 기본 구조 이해 (정방향/역방향 과정)
- [ ] MSE loss를 통해 노이즈 예측 학습 원리 파악
- [ ] U-Net 구조 이해 (denoising backbone으로 사용됨)
- [ ] timestep embedding 방식 학습
- [ ] Sampling 방식 비교: DDPM vs DDIM vs DPM-Solver

### 💡 추천 학습자료
- 논문: *Denoising Diffusion Probabilistic Models (Ho et al., 2020)*
- 코드 실습: Huggingface `diffusers` 라이브러리의 `DDPMPipeline`
- 과제: MNIST 또는 CIFAR-10으로 DDPM 직접 구현해보기 (U-Net 사용)

---

## ✅ 2단계: 조건부 Diffusion (Conditional Generation)

**→ 조건(condition)을 주어 원하는 특성의 이미지를 생성하도록 확장**

- [ ] Class-conditional generation (ex. 숫자 라벨로 MNIST 생성)
- [ ] Text-conditional generation (텍스트로 이미지 생성)
- [ ] Cross-attention 기반 conditioning 방식 이해

### 💡 추천 논문/모델
- [ ] Classifier-Free Guidance (CFG)
- [ ] GLIDE (OpenAI): Text-to-Image diffusion 도입
- [ ] Stable Diffusion: Latent 공간에서 조건부 diffusion 방식

---

## ✅ 3단계: Text-to-Image Diffusion (Stable Diffusion 중심)

- [ ] Text Encoder (CLIP/Text Transformer) 구조 및 역할 이해
- [ ] Latent Diffusion Model (LDM) 이해 → 효율적 고해상도 생성
- [ ] Cross-attention으로 텍스트 정보를 반영하는 방식 익히기
- [ ] Classifier-Free Guidance로 텍스트 조건 반영 강도 조절하기

### 💡 추천 리소스
- 논문: *High-Resolution Image Synthesis with Latent Diffusion Models*
- 실습: HuggingFace `StableDiffusionPipeline` 사용해보기
- 과제: prompt에 따라 생성 이미지가 어떻게 변하는지 실험해보기

---

## ✅ 4단계: Image-to-Video (Video Generation) 확장

**→ 이미지 생성에서 시간축을 추가하여 영상 생성으로 확장**

### ▶ 주요 방식
- [ ] Frame-by-frame 생성: 시간에 따라 이미지 하나씩 생성
- [ ] Spatiotemporal diffusion: 시간+공간 함께 모델링
- [ ] Video-LDM: Latent 공간에서 영상 생성하는 방식

### ▶ 학습 주제
- [ ] Temporal consistency (시간 흐름 일관성) 유지 방법
- [ ] 3D U-Net 또는 Time embedding 구조 학습
- [ ] First-frame conditioning / Text + Motion 제어 방법

### 💡 추천 논문/모델
- [ ] *Video Diffusion Models* (Ho et al., 2022)
- [ ] *Imagen Video* (Google)
- [ ] *VideoCrafter*, *ModelScope*, *Gen-2* (Runway ML)

