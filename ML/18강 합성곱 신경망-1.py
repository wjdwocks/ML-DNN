from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 이번에는 -1, 28, 28, 1로 크기를 재정의함.
# 즉, (28, 28, 1)의 크기인 사진이 48000개 있게 되는것임. 뒤의 깊이 1을 추가해줌.
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0 # 즉, (48000, 28, 28, 1)의 4차원 배열이 됨.
# 여기에서 맨 처음 -1은 48000개 사진을 의미하는데 이를 배치 차원이라고 함.

from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))
# 여기서 첫 번째 매개변수는 필터의 개수, kernel_size는 이 필터의 크기를 의미하는데, 3이면 3 x 3필터를 의미하고, 5이면 5 x 5 필터를 의미한다.
# 여기서 depth차원을 따로 지정해주지 않았는데, 이는 원본의 깊이를 따라가기 때문이다.
# 즉, 필터의 가로 세로의 크기만 3으로 정해줬는데 깊이는 원본을 따라가기 때문이라는 거임.

## 위의 필터 32개를 거친 후에 생기는 특성 맵은 (28, 28, 32)의 크기가 됨. 왜?) padding='same'으로 주었기 때문에.
# 위의 input_shape에는 배치 차원이 들어가지 않고, 각 사진 당 어떤 형태를 띄는지만을 써줌.
model.add(keras.layers.MaxPooling2D(2)) # MaxPooling이 좀 더 선호된다.
# 풀링은 우리가 위의 합성곱 연산에서 전체 입력을
# 여기서 MaxPooling2D는 2차원 풀링을 의미하고, 2는 풀링에서의 필터 크기이다.
# 풀링
# 여기서도 depth차원은 그대로 유지된다.


### 위에까지가 첫 번째 합성곱 층 + 풀링 층을 만든 것임.
# 여기서 두 번째 합성곱 층 + 풀링 층 + 완전연결 층으로 마무리를 해보자
# 전의 층에서 (14, 14, 32)의 크기의 특성 맵이 형성되어 이번 층의 입력으로 왔을 것임.
# 이번에는 64개의 (3, 3)크기의 필터를 가지고 합성곱 층을 만들어보자.
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
# 위의 코드를 통해서 (3, 3, 32)의 크기를 가진 필터 64개로 합성곱 연산을 수행하여 새로운 특성 맵을 만든다.
# 이렇게 한 결과로는 (14, 14, 64)의 크기가 될 것임. 왜?) padding='same'이었기 때문.
model.add(keras.layers.MaxPooling2D(2))
# MaxPooling2D로 특성 맵을 (7, 7, 64)로 변경함.

### 여기서부터는 완전연결층으로 변경하는 과정.
model.add(keras.layers.Flatten()) # Flatten()층을 추가하여 (7, 7, 64)의 입력을 3136개의 1차원 입력으로 변경함.
model.add(keras.layers.Dense(100, activation='relu')) # 3136개의 입력을 100개의 뉴런으로 줄여줌. 활성화함수 'relu'를 사용.
model.add(keras.layers.Dropout(0.4)) # 이 때 과대적합을 막기 위해서 Dropout층을 추가해준다. (0.4)의 의미는 각 배치당 0.4배의 비율만큼 비활성화한다는건데
# Dropout에 대해서는 난중에 다시 공부해보자. 아직 인공신경망 모델이 어떻게 흘러가는지도 확신이 안서네.
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

### 실제로 위에서 만든 모델을 학습시켜보자.
# optimizer는 'adam'으로, 손실함수는 크로스엔트로피로, 훈련 각 epoch당 accuracy를 출력하도록 metrics를 지정해줌.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# callback을 사용하여 조기종료를 구현할 것임. 
# checkpoint를 사용하여 가장 loss가 낮았을 때의 모델을 best_cnn_model.h5에 저장함.
checkpoint_cb = keras.callbacks.ModelCheckpoint('best_cnn_model.keras')
# 조기 종료를 위해서 loss가 계속 감소하다가 증가하게 된다면(2번의 유예를 줘서 다시 내려가면 기다림) 증가하기 전의 epoch를 위의 best_cnn_model.h5에 저장.
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

# fit을 하는데, 기본 epoch는 20, 검증 데이터를 넘겨줘서 검증 세트도 학습을 하여 그 정보를 history객체에 넣어줌.
history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

### 모델의 평가와 예측
model.evaluate(val_scaled, val_target)

import matplotlib.pyplot as plt
plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()

preds = model.predict(val_scaled[0:1])
print(preds)

plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
model.evaluate(test_scaled, test_target)