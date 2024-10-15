from tensorflow import keras

model = keras.models.load_model('머신러닝/best_cnn_model.keras')
print(model.layers)
# 이전 장의 callback을 이용하여 최적의 epoch에서 조기종료된 모델을 불러옴.

### 합성곱 신경망층의 가중치와 절편을 살펴보자.
conv = model.layers[0] # 위의 모델의 첫 번째 층(합성곱 모델)을 객체로 받음.
print(conv.weights[0].shape, conv.weights[1].shape)
# 그 첫 번째 층의 가중치는 (3, 3, 1, 32)의 형태, 절편은 (32,)의 형태를 띔.
# 그 이유는 필터(커널)의 크기가 (3, 3)이고, 깊이가 1이기 때문에 (3, 3, 1)의 필터였고,
# 그 필터를 32개 만들었기 때문임.
# 절편의 경우에는 각 필터 32개마다 1개의 절편을 가지고 있기에 (32,)의 크기가 된다.

### 합성곱 신경망층의 가중치와 절편의 평균과 표준편차는?
conv_weights = conv.weights[0].numpy() # 위의 conv.weights는 tensor클래스의 객체이다.
# numpy로 알아보기 위해서 .numpy()를 해서 numpy배열로 바꾸어준다.
print(conv_weights.mean(), conv_weights.std())
# 평균은 0에 가깝고, 표준편차는 0.24정도가 나옴.
# 히스토그램을 이용해서 가중치들의 분포를 살펴보자.
import matplotlib.pyplot as plt
plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show() 

## 32개의 필터를 출력해보자. 이게 각 필터들의 가중치를 시각화해서 그린 것임.
fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[:, :, 0, i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
        # 32개의 사진이 있어서 각 행에 16개씩 2개의 열에 그림을 배치하려고 함.
        # axs[i, j]번째 위치에 필터의 크기 (:, :) == (3, 3) 의 깊이 0(1) 의 사진을 두는 것임.
        # 즉, i, j번째 위치에 3x3 크기의 사진을 배치함.
plt.show()    
# 그림을 보면 어떻게 학습이 되었는지는 잘 알진 못하고, 그냥 밝은 쪽이 가중치가 높은 것을 의미한다.

## 이번에는 학습되지 않은 모델의 가중치를 살펴보자.
no_training_model = keras.Sequential() 
no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))
# 모델을 생성하고, 합성곱 신경망을 추가함.
no_training_conv = no_training_model.layers[0]
print(no_training_conv.weights[0].shape)
# 합성곱 신경망을 똑같이 추가했지만, 뒤는 추가하지 않고, 학습도 돌리지 않음.
no_training_weights = no_training_conv.weights[0].numpy()
# 그 때의 가중치(학습시키지 않은)를 가져와서 히스토그램으로 그려봄.
print(no_training_weights.mean(), no_training_weights.std())

plt.hist(no_training_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()
# 가중치가 전에는 종모양으로 보였다면 여기서는 균등 분포의 모습을 띈다.

fig, axs = plt.subplots(2, 16, figsize=(15, 2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(no_training_weights[:, :, 0, i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()
# 그림이 전체적으로 어둡다. (밝은 쪽이 적은 것 같다.)

### 함수형 API로 모델을 만드는 방법.
# 지금까지는 Sequential클래스를 이용해서 모델을 만들었다.
# 하지만 딥러닝에서의 복잡한 모델들을 만들기에는 부족하다. 입력이 2개, 출력이 2개인 경우도 있다.
# 이럴 때 함수형 API를 이용할 수 있다.

# 모델에서 사용할 레이어를 정의함.
# dense1 = keras.layers.Dense(100, activation='sigmoid')
# dense2 = keras.layers.Dense(10, activation='softmax')

# 입력 텐서를 정의함.
# inputs = keras.Input(shape=(784,))

# 레이어들을 연결해준다.
# hidden = dense1(inputs)
# outputs = dense2(hidden)
# model = keras.Model(inputs, outputs)

# inputs = keras.Input(shape=(784,))  
## 이런 식으로 이을 수 있다는 것이다.

# inputs -> hidden(dense1) -> outputs(dense2) -> model(input, output)이 됨.



### 이제 우리가 전 장에서 만든 Sequential클래스의 모델을 보면
# model = input Layer -> Conv2D -> Maxpooling2D -> Conv2D -> MaxPooling2D -> Flatten -> Dense -> Dropout -> Dense(output)의 형태이다.
# 함수형 API를 이용하면 앞의 Conv2D의 입력과 출력을 알 수 있게 되어 새로운 모델을 얻을 수 있게 된다.
# model 객체의 predict() 메서드는 처음 입력부터 마지막 층까지 모든 계산을 수행한 뒤 최종 출력을 반환하지만,
# 우리가 필요한 첫 번째 Conv2D를 지난 후 출력된 특성맵이다. 이 출력은 Conv2D객체의 output속성에서 얻을 수 있다.)
print('얘가 입력임', model.input) # 얘가 처음 입력임.
# model.layers[0].output이 첫 번째 합성곱층을 지난 후 출력된 특성 맵이다.
conv_acti = keras.Model(model.input, model.layers[0].output) # 이것이 처음 입력과 처음 Conv2D를 지난 후의 출력으로 만든 새로운 모델임.
# 위의 conv_acti도 model객체이므로 이 conv_acti.predict()를 하면 첫 번째 conv2D만을 지난 후의 결과를 예측할 것임.

## 첫 번째 그림에 대해서 위의 conv_acti 모델로 예측해보자.
# 첫 번째 train_input[0]그림을 그려보자.
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
plt.imshow(train_input[0], cmap='gray_r')
plt.show()

# 그 그림을 떼와서 전처리를 해준 뒤 predict를 해봄.
inputs = train_input[0:1].reshape(-1, 28, 28, 1) / 255.0

feature_maps = conv_acti.predict(inputs)

# 세임 패딩이고, 32개의 필터를 사용했기에, (28, 28, 32)의 크기가 나옴. 첫 번째 차원은 배치 차원임.(샘플이 1개였어서 1임.)

fig, axs = plt.subplots(4, 8, figsize=(15, 8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
        axs[i, j].axis('off')
plt.show()