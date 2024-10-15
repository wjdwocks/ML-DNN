from tensorflow import keras
import numpy as np

# tensorflow의 keras에서 패션 Mnist 데이터를 불러올 수 있다.
# keras.datasets에서 데이터를 로드 할 때에는 두 쌍의 값을 반환하는데,
# (학습 데이터, 학습 타겟), (테스트 데이터, 테스트 타겟)의 형태로 반환한다.
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

print(train_input.shape, train_target.shape) # 학습 데이터로 60000개의 데이터가 옴
print(test_input.shape, test_target.shape) # 테스트 데이터로 10000개의 데이터가 옴.

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 10, figsize=(10, 10)) # 매번 봐도 익숙하지 않은데, matplotlib.pyplot의 subplots은 여러 사진을 병렬로 한 figure에서 보기 위함.

for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r') # image show의 줄임으로, train_input[i]에 2차원 배열이 있을 텐데 그 값들을 조합해서 이미지를 생성해준다.
    axs[i].axis('off') # axs는 위에서 1, 10이었으므로 1행 10열 중 그 위치를 특정해줌.
plt.show()
print(train_target[:10])

print(np.unique(train_target, return_counts=True)) # train_target에서 각 값(1 ~ 10)까지 몇개씩 있나를 보여줌. → unique함수 역할

train_scaled = train_input / 255.0 # 각 픽셀들 값이 0~255라면 값이 너무 커지게 되어서 값을 줄여주면 좋다. (정규화의 느낌)
train_scaled = train_scaled.reshape(-1, 28*28) # 특성을 2차원이 아닌 1차원 특성으로 변환해준다. SGD는 1차원의 형태로 받음.
print(train_scaled.shape) # (60000, 784) 60000개의 데이터가 각각 784개의 특성을 가진다.

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
sc = SGDClassifier(loss = 'log', max_iter=5, random_state=42) # 손실험수를 log로 줌.
# 다중 분류문제인데 loss='log'인 이유? 
# 원래 이중분류가 logistic 함수, 다중 분류는 ~함수로 사용함.
# 근데 sklearn에서 그냥 다중 분류 모델에 log를 주면, 각 타겟이 되는 클래스(여기서는 10개)를 양성, 나머지를 음성으로 줘서 각각의 값을 얻고, 이를 소프트맥스 함수로 확률로 변환한다.
# 이중 분류는 시그모이드 함수.

scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score'])) # 이름은 test점수라고 하지만, 검증 점수이다. 테스트 데이터는 사용하지 않았다. 5-fold 교차 검증 점수의 평균임.
# 로지스틱 회귀에서는 z_1(티셔츠) = w_11(티셔츠에 해당하는 1번픽셀의 가중치) * z_1(1번 픽셀) + w_12 * z_2 + ... + b 로 나타남.
# 즉, 각 티셔츠, 바지 등등 분류될 클래스에 각각 픽셀에 해당하는 가중치가 다 달라야 함.


### Keras를 이용한 딥러닝 모델만들기
#1. 케라스 모델 만들기
# 딥러닝에서는 교차 검증을 잘 사용하지 않는다. (위의 cross_validate같은 것) 왜냐하면, 데이터가 충분히 많기에 테스트 점수가 안정적이기도 하고, 교차검증을 하려면 모델을 여러개 만들어서 계속 돌려봐야하는데 그럴 시간이 없다.
# 그래서 오리지널 식으로 학습 데이터를 일부 떼와서 검증 데이터로 사용하는 방식으로 함.
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42) 

print(train_scaled.shape, train_target.shape)
print(val_scaled.shape, val_target.shape)

# 가장 기본이 되는 층부터 시작함. 
#2. dense 층 (완전 연결층)
# 10개의 뉴런(출력값), 과 여러 개의 특성들(입력층)이 모두 곱해지면서 가중치들이 생겨서 이를 선으로 연결하면 아주 뺵빽하게 선이 그어지므로 밀집층(dense)라고 생각하라.
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,)) 
# 분류 문제이기 때문에, 10개의 클래스가 있으니 10개의 뉴런(유닛)을 지정해주어야함.
# 다중분류이기 때문에 softmax함수를 지정해 준다. 
# dense층에서 각 특성에 대해? z값들을 10개 계산해서 softmax로 확률을 출력해줌.
# 첫번째 모델에 추가되는 층에는 input_shape을 지정해줘야 함. (나중에 모델을 만들 때 편하다.)
model = keras.Sequential(dense) # 입력, dense, 출력층 모두를 포괄하는 모델인것임. ????? 출력층 하나만 포함되어있다 여기서는?

#3. 모델 설정(compile)
# compile 메소드에서는 손실함수를 설정해주고, 
# 이진 분류일 때는 loss=binary_crossentropy
# 다중 분류일 때는 loss=categorical_crossentropy
# 여기서 앞의 sparse는 왜붙냐면 원 핫 인코딩(하나만 남기고 나머지는 0으로 변환)하기 위해?
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

# 실제로 학습을 돌릴 때에는 머신러닝과 같이 fit함수를 사용함.
model.fit(train_scaled, train_target, epochs=5) # epochs : 반복 횟수를 정해줄 수 있음.

# 검증을 직접 해줌.
model.evaluate(val_scaled, val_target) # 검증 점수.





