## Baseline으로 사용할만 한 다양한 Multi Modal KD 기법들 정리
### 1. AVER (Averaged KD)

* 여러 Teacher의 logit 출력을 단순 평균한 후, Student가 이를 학습하도록 유도
* 특징: 구현이 단순하고 baseline 성격이 강함
* Sig, GAF, PI의 Teacher logit을 평균하여 사용 가능

```python
avg_teacher_logits = (logits_sig + logits_gaf + logits_pi) / 3
loss_kd = KL(student_logits, avg_teacher_logits)
```

### 2. ESKD (Early Stopped KD)

* 학습을 할 때 Acc기준으로 좋아졌을 때의 모델을 저장하고, 가장 좋았던 시점의 모델을 사용
* Full KD는 항상 지정한 Epoch이 끝난 후의 모델을 사용하여 과적합 상태가 될 수 있음

### 3. CAMKD (Class Activation Map Knowledge Distillation)

* CNN 기반 Teacher와 Student 간의 CAM을 비교하여, 시각적 근거를 Distillation해줌.
* GAF, PI는 이미지 기반이므로 CAM 생성 가능
* Student의 이미지 branch가 존재해야 함
* 코드가 있는지 모르겠다.

```python
feat_t_gaf = teacher_gaf.get_feature(x_gaf).detach()
feat_s_gaf = student.get_feature_gaf(x_gaf)
cam_t = get_cam(feat_t_gaf, teacher_gaf.fc[y]) # 여기서 CAM을 얻어내고
cam_s = get_cam(feat_s_gaf, student.fc[y])
loss_cam = MSE(cam_t, cam_s) # 여기서 둘을 MSE로 비교해서 Loss 계산
```

전체 Loss 예시:

```python
loss_total = lambda1 * ce_loss + lambda2 * KD_loss + lambda3 * cam_loss
```

### 4. EBKD (Evidence-Based Knowledge Distillation)

* Teacher의 판단 근거 (evidence, CAM 또는 attention map 등)를 Student에게 전달
* Cross-modal하게 evidence를 통합하여 비교하는 것도 가능

```python
attn_sig = teacher_sig.get_attention(x_sig).detach()
cam_gaf = teacher_gaf.get_cam(x_gaf).detach()
student_attn = student.get_cross_attention(x_sig, x_gaf)
loss_ebkd = MSE(student_attn, fuse(attn_sig, cam_gaf))
```

### 5. Enhancing Time Series Anomaly Detection: A KD Approach (2024)

* 시계열 이상탐지 전용 KD 기법
* Logit 대신 hidden feature를 직접 Student에 전달
* 일반화 성능 향상 목적, 특히 Sig branch 학습에 유리함

```python
loss_feature_kd = MSE(student.hidden_vec, teacher_sig.hidden_vec.detach())
```

### 6. Progressive Cross-modal KD for HAR (2022)

* Human Activity Recognition(HAR) 특화 Cross-modal KD
* 점진적 전이 구조: Sig -> Image, Image -> Sig, 최종 shared representation 학습
* 각 단계별로 Teacher → Student 간 feature 전이

```python
loss_kd_sig2img = MSE(student_img_feature, teacher_sig_feature.detach())
loss_kd_img2sig = MSE(student_sig_feature, teacher_img_feature.detach())
```

### Frequency Attention for Knowledge Distillation, 2024.3.9

---

## 📘 논문 요약

- 본 논문은 Knowledge Distillation(KD)을 수행할 때, **주파수 도메인(Frequency Domain)** 에서 Attention을 적용하여 Teacher의 전역적인 정보를 효과적으로 Student에게 전달하는 기법을 제안한다.
- 특히, 공간 도메인(spatial domain)이 아닌 **Fourier 주파수 도메인에서 Attention을 수행**하여, 더 넓은 문맥 정보(엣지, 패턴, 반복 구조 등)를 포착하는 것이 핵심 아이디어다.
- 이러한 방식은 기존 KD에서 흔히 사용되는 MSE 또는 feature mimic 방식보다 더 효과적으로 동작하며, 다양한 Teacher/Student 구조에 적용 가능하다.
- 여기에서 나는 Signal, Image의 멀티모달 방식이므로, Feature를 (batch, Channel, Window) -> (batch, Channel, Window, Window) 에서, Window 크기를 Image의 Width와 Height로 맞춰주면 적용할 수 있을 것으로 보인다.

---
## 공간 도메인과 주파수 도메인
- 주파수란 : 어떤 신호에서 얼마나 빠르게 값이 바뀌는지를 나타내는 개념이다.
- 느린 변화 : 배경, 큰 물체의 윤곽 등 -> 저주파
- 빠른 변화 : 엣지, 텍스처, 작은 디테일 등 -> 고주파
- 공간 도메인 : 원본 이미지, 시계열의 직접적인 픽셀 값
- 주파수 도메인 : 이미지, 시계열의 구성 성분을 주기적 신호들로 분해한 값
- Fourier 변환이란 : 신호를 여러 개의 사인파의 합으로 분해하는 것.
- 이 Fourier 변환을 이용해서 중간 Layer의 Feature Map을 주파수 도메인에서의 Feature Map으로 변환하고, Teacher와의 Attention을 수행하고 Student를 변환한 뒤, Teacher와의 주파수 도메인에서의 Feature Map과 MSE를 통해 Loss를 계산한다.

---

## 🧠 주요 구성 요소

### 1. Frequency Attention Module (FAM)

- FAM은 주파수 도메인으로 변환한 feature에 **학습 가능한 필터를 적용**하여, Student의 feature를 Teacher의 방향으로 정렬(유도)하는 역할을 한다.
- 구성 흐름:
  1. `torch.fft.fft2()`를 사용해 입력 feature를 주파수 도메인으로 변환
  2. 복소수 필터(`weights1`)를 이용한 학습 가능한 convolution 연산 수행
  3. 중심 주파수 제거(Masking)로 고주파 강조
  4. `ifft2()`로 다시 공간 도메인으로 복원
  5. 보조 분기(`1x1 conv`)와 조합하여 최종 attention feature 생성

---

## 🔍 코드 기반 동작 흐름

```python
x_ft = torch.fft.fft2(x, norm="ortho")  # FFT2D 수행
out_ft = self.compl_mul2d(x_ft, self.weights1)  # 학습 가능한 복소수 필터 곱
batch_fftshift = batch_fftshift2d(out_ft)  # 중심 이동
# 중심(low freq) 마스킹
batch_fftshift[:, :, cy-rh:cy+rh, cx-rw:cx+rw, :] = 0
out_ft = batch_ifftshift2d(batch_fftshift)
out_ft = torch.view_as_complex(out_ft)
out = torch.fft.ifft2(out_ft, norm="ortho").real  # 복원
```


### CRD - 2024, 