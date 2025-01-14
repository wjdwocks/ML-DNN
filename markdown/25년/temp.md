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