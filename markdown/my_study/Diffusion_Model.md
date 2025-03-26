### Diffusion Model에 대해 공부해 볼 것이다.
- 얘는 아마도 생성형 AI쪽에 관련한 얘인것 같다.
- Denoising Diffusion Probabilistic Models 논문을 참고해서 공부할것임.
- GPT를 통해 Diffusion Model의 기본 개념을 공부한 뒤, 논문을 읽어보며 이것의 깊은 이해와 모델의 활용 등을 이해해보자.


### Diffusion Model의 기본 개념
1. Forward Process : 원본 이미지에 점진적으로 노이즈를 추가해서, 완전한 노이즈로 만드는 과정. (학습에 사용된다.)
2. Reverse Process : 완전한 노이즈로부터, 점차 노이즈를 제거하여 원본 이미지를 복원하는 과정. (신경망은 이것을 학습한다.)
3. 학습 목표 : 각 단계에서 '추가된 노이즈'를 예측하여 제거하는 것. (결국, 이 Model은 Denoising 능력을 학습하게 됨.)
4. 생성 시나리오 : 랜덤한 노이즈를 입력으로 넣고, 학습된 모델로 Denoising을 반복함 → 결국 새로운 이미지가 생성되게 됨.


### Diffusion Model의 학습 흐름.
1. 원본 이미지에 랜덤한 노이즈(ϵ)를 추가하여 Model에 Input으로 넣는다. 아레의 x_t가 모델의 입력이 된다.
<ul>
- 시간 step t를 랜덤으로 고른 뒤, 해당 step에 해당하는 노이즈 비율에 따라 정해진 공식으로 노이즈를 섞은 x_t를 만든다. ϵ∼N(0,I)
</ul>
<img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/Study/Diffusion_Model/step1.png" alt="Step1" width="700">

2. 모델은 주어진 noisy image x_t 에서 해당 시점 t에 섞여 있던 노이즈 ϵ를 예측한다.
<ul>
- 모델의 parameter를 거쳐서, 노이즈 ϵ를 예측하게 됨.
</ul>
<img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/Study/Diffusion_Model/step2.png" alt="Step2" width="700">

3. 모델의 loss 함수는 노이즈 ϵ를 얼마나 잘 예측했는지를 계산한다.
<ul>
- 모델이 예측한 노이즈 ϵ와 우리가 실제로 섞은 노이즈 ϵ_t 간의 차이를 loss로 사용한다.
</ul>
<img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/Study/Diffusion_Model/step3.png" alt="Step3" width="700">

4. 모델은 랜덤한 노이즈를 넣었을 때 원본에 가깝게 Denoising하는 능력을 학습하게 된다.
<ul>
- 모델이 시간 step t 마다 노이즈를 얼마나 잘 제거할 수 있는지 학습한다.
- 이걸 반복해서 모델은 완전한 노이즈에서 점점 더 원본처럼 보이도록 복원하는 Reverse Process를 더 잘 수행할 수 있게 됨.
</ul>

5. 이게 왜 Denoising Model이 아니라, Generative Model인지
<ul>
- 학습 과정에서 본 이미지를 복원하는 것이 목적이 아님.
- 노이즈 제거 능력을 일반화 하는 것이 목표이다.
- 학습이 완료된 모델은 완전히 랜덤한 노이즈 x_t를 넣더라도, 그것을 잘 Denoising하여 그럴 듯한 새로운 이미지를 출력하게 된다.
</ul>