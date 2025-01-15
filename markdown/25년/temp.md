### RNN
<li> Sequence 데이터를 처리하기 위한 신경망 알고리즘. </li>
<li> 1. 입력 데이터를 입력 Sequence(Vector)로 표현하여 넣어줘야 한다. </li>
<li> 2. 아레와 같이 W, V, U 행렬들이 우리가 학습해야 할 파라미터이다. </li>
<li> 3. W, V, U의 초기값은 랜덤으로 설정이 될 것이고, 경사하강법을 통해서 업데이트 됨. </li>
<li> 4. h는 hidden state로, 현재까지의 Sequence 정보를 축적하고 있는 상태이다. RNN의 시간축을 따라 순차적으로 업데이트되고, 이전 hidden state(h_{t-1})와 현재 입력(x_t)를 결합하여 계산한다. </li>
<li> 5. o는 output vector로, Hidden State를 바탕으로 계산된 출력 값으로, 모델의 최종 예측값을 얻기 위한 중간 단계이다. o_t는 hidden state h_t를 기반으로, 현재 time step에서 모델이 생성한 가중치 기반의 해석이다. </li>
<li> 일단 이렇게 알고만 있으라는디 </li>
 
 ![RNN](https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/25.1.16/RNN.png)

 ### LSTM
 <li> 기존 RNN은 장기 의존성이라는 문제가 있었다. </li>
 <li> 장기 의존성 문제란? : time sequence가 점점 길어진다면, 뒤로 갈 수록 chain rule에 의해 계산되는 식이 0에 수렴한다는 것이다. </li>

  ![장기 의존성 문제](https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/25.1.16/long-term.png)
  <li> 장기 의존성 문제의 사례 : '당신의 내면의 힘을 과소평가하지 마세요' 를 번역할 때 'Don't underestimate your inner strength'로 해석이 될텐데, 마세요는 마지막이지만, Don't는 첫 번째 time step이기 때문에, 서로 의미적으로 가까운 단어이지만, Long Term dependency에 의해 학습이 잘 되지 않을 것이다. </li>
  <li> LSTM이 기존 RNN과 다른 점 : Gate Algorithm(forget gate, input gate, candidate gate, output gate) </li>
  <li> forget gate : 이전 Hidden State와 현재 입력(x)를 받아서 Forgot Gate를 통해 잊을 것을 파악한 뒤 이전 Cell State에서 잊을 정보를 지운다. </li>
  <li> Input Gate : 이전 Hidden State와 현재 입력 x를 받아서 Input Gate를 통해 기억할 것을 파악한다. </li>
  <li> Candidate Gate : 값을 -1 ~ 1 사이의 값으로 정규화 시키는 역할을 하며, Input Gate와 곱하여서 Cell State에 기억할 정보를 추가하도록 한다. </li>
  <li> Output Gate : 이전 Hidden State와 현재 입력 x를 받아서 Output을 출력하고, 업데이트된 Cell State와 결합하여 다음 Hidden State를 생성하는 역할을 한다. </li>
  <li> 즉, Gate 알고리즘으로 Cell State(장기기억)을 관리하고, Hidden State(단기 기억)은 그대로 또 유지하는 것이 LSTM이다. </li>

  ![게이트 알고리즘](https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/25.1.16/gate_algorithm.png)


### Sequence To Sequence 알고리즘
<li> 이 알고리즘은 Sequence to Sequence Learning with Neural Networks 논문에서 처음 제안되었다. </li>
<li> 기계 번역에서의 주요 문제 중 하나인 장기 기억(long-term dependency)문제를 해결하기 위해 LSTM을 기본 단위로 사용한다. </li>
<li> 또한, 기계 번역에서의 두 번째 주요 문제인 문장 내 어순 및 단어 개수의 불일치를 해결하기 위한 접근 방식으로 도입되었다. </li>
<li> 기본 구조 </li>
<ul>
<li> Seq2Seq 모델은 LSTM을 기반으로 설계된 Encoder-Decoder 구조로 입력 시퀀스를 처리한다. </li>
<li> Encoder : 입력 Sequence를 처리하며 최종 Hidden State를 생성한다. </li>
<li> Context Vector : 마지막 LSTM의 Hidden State를 요약된 정보로 사용하는데, 이를 Context Vector라고 한다. </li>
<li> Decoder : Context Vector를 초기 상태로 받아, 번역된 문장을 생성한다. (Encoder와 Decoder는 당연히 서로 다른 가중치 파라미터를 사용한다.) </li>
</ul>
<li> 개선될 부분 : Seq2Seq 모델은 Context Vector(Encoder-Decoder 사이의 Cell/Hidden State)가 고정된 크기라는 점에서 입력 Sequence가 긴 경우 정보 손실이 발생할 수 있다는 문제가 있다. </li>
<li> 내 생각(정리 해보자.) </li>
<ul>
<li> Encoder 부분도, Decoder 부분도 모두 LSTM으로 되어있다. </li>
<li> Encoder 부분에서는 출력을 내는 대신에, Hidden State와 Cell State만 반복해서 업데이트를 수행한다. </li>
<li> 그렇게 되면 마지막 Hidden State와 Cell State는 문맥에 대한 모든 정보가 들어가게 됨. 이것을 Context Vector라고 한다.(Context Vector = Cell State + Hidden State) </li>햐햐
<li> Decoder 부분에서는 이 Context Vector와 SOS(Start of Sentence)의 초기값을 이용하여 단어를 출력하게 되고, 이전 Hidden State와 Cell State를 이용하여 이번 LSTM(time step)의 Cell State와 Hidden State를 업데이트하여 단어를 출력하는 것을 반복한다. </li>
<li> 여기에서 입력 sequence와 출력 sequence의 길이가 달라질 수 있는 이유는 EOS가 나올 확률이 가장 높아질 때까지 단어 생성을 반복하기 때문이다. </li>
</ul>

![Seq2Seq](https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/25.1.16/Seq2Seq.png)


### Attention Model



### Self-Attention



### Transformer


