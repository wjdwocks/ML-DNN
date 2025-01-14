### RNN
<li> Sequence 데이터를 처리하기 위한 신경망 알고리즘. </li>
<li> 1. 입력 데이터를 입력 Sequence(Vector)로 표현하여 넣어줘야 한다. </li>
<li> 2. 아레와 같이 W, V, U 행렬들이 우리가 학습해야 할 파라미터이다. </li>
<li> 3. W, V, U의 초기값은 랜덤으로 설정이 될 것이고, 경사하강법을 통해서 업데이트 됨. </li>
<li> 4. h는 hidden state로, 현재까지의 Sequence 정보를 축적하고 있는 상태이다. RNN의 시간축을 따라 순차적으로 업데이트되고, 이전 hidden state(h_{t-1})와 현재 입력(x_t)를 결합하여 계산한다. $ h_t = tanh(Wh_{t-1} + U x_t) $ </li>
<li> 5.  </li>
<li>  </li>
 
 ![RNN](https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/25.1.16/RNN.png)