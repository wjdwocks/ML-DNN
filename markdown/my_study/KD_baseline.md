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