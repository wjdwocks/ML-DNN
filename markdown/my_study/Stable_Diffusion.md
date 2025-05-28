# Stable Diffusion 공부
## Stable Diffusion
    * 텍스트를 입력으로 받아 이미지를 생성하는 고품질 조건부 생성 모델
    * 기존의 Diffusion 구조를 Latent 공간에서 더 효율적이고, 안정적으로 작동하도록 재설계한 것.

## 왜 Stable 인가?
    * 학습 안정성 향상
        - Full-Resolution 이미지 대신 Latent Space에서 학습해서 안정적이다.
        - 기존 Diffusion에서는 직접 RGB 이미지에 노이즈를 추가하고, 다시 원본 이미지로 복원하게끔 학습이 이루어짐
        - Stable Diffusion은 원본 이미지를 VAE로 latent vector로 변환해서 그 latent 공간에서만 Diffusion을 수행한다.
        - 즉, VAE로 입력 이미지를 latent space z로 변환하고, 이 z에 노이즈를 추가. 그 후 z_T를 보고 z와 비교하여 노이즈를 예측하게 된다.
    * 메모리 안정성
        - 원본 이미지를 latent로 압축 하기에 학습/생성 메모리가 훨씬 작다.
    * 결과의 의미론적 안정성
        - CLIP-based Conditioning을 통해 텍스트에 맞는 일관된 이미지를 생성할 수 있음.

## 학습 과정
    * VAE Encoder
        - RGB 이미지를 VAE를 통해 Latent Vector로 압축한다.
    * Forward Process (노이즈 추가)
        - 시간 t에 대해서 일정한 스케줄에 따라 z_0에 노이즈를 섞어 z_t를 만듦.
    * Text Prompt Encoding (요게 중요함...)
        - 텍스트 프롬프트 -> CLIP Text Encoding -> Text Embedding c를 추가함.
        - 이 텍스트 임베딩도 학습이 되어야 하며, latent vector z에 c가 더해지는(?) 방식으로 진행됨.
    * U-Net 학습
        - z_t, t, c를 보고 노이즈 ϵ를 예측함.
    * Loss 계산
        - |ϵ(z_t, t, c) - ϵ|² 으로 예측 Loss를 사용하여 계산한다.

## Text Embedding은 어떻게 만들어지나
    * 텍스트 임베딩은 CLIP의 Text Encoder를 사용해서 만들게 됨.
    * 혹은 OpenAI의 ViT-B/32 + Transformer 구조 기반 학습
    * 즉, 이미 학습된 Text Embedding 을 통해 수행한다고 하는데, 학습은 Vision Transformer에서 Embedding을 학습하던것처럼 하나봄.

## Text Embedding이 어떻게 Latent Space z에 추가되나
    * z와 Text Embedding이 Cross Attention을 통해서 결합됨.
    * Query가 z
    * Key, Value가 Text Embedding
    