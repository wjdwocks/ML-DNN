from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

from sklearn.model_selection import train_test_split
train_scaled = train_input/255.0
train_scaled = train_scaled.reshape(-1, 28*28)
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)


### 은닉층(학습 할 층) 두개 만들어서 학습하기
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)) # 입력층 다음으로 처음 실행될 은닉층
dense2 = keras.layers.Dense(10, activation='softmax') # 출력을 하기위한 층.

model = keras.Sequential([dense1, dense2]) # 마지막에는 출력층이 와야하고, 처음에는 입력층 다음으로 수행할 은닉층이 와야 함.

# model.summary()

### Sequential의 생성자에 직접 Dense의 생성자를 넣기
model = keras.Sequential([keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'), keras.layers.Dense(10, activation='softmax', name='output')], name='패션 MNIST 모델')
# model.summary()
# add()메서드를 이용한 Sequential 객체에 층 추가하기
model = keras.Sequential()
model.add(keras.layers.Dense(300, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dense(100, activation='relu', input_shape=(300,)))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

# 모델의 훈련
# model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_scaled, train_target, epochs=5)

# Flatten을 이용한 모델 + ReLU활성화 함수
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input/255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, random_state=42, test_size=0.2)
model.fit(train_scaled, train_target, epochs=5)

# 검증 세트의 평가
model.evaluate(val_scaled, val_target)


### 옵티마이저
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 여기서의 sgd는 sgd = keras.optimizers.SGD()와 같다.
sgd = keras.optimizers.SGD()
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 위에서 SGD의 learning_rate 매개변수를 바꿔줄 수 있다.


### 옵티마이저 적용 예시
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_scaled, train_target, epochs=10)

model.evaluate(val_scaled, val_target)








