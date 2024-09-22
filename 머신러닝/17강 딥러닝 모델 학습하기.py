from tensorflow import keras
from sklearn.model_selection import train_test_split


def model_fn(a_layer=None): # 저번 장에서 만들었던 기본적인 모델을 형성하는 함수
    model = keras.Sequential() # 모델의 틀을 만들고
    model.add(keras.layers.Flatten(input_shape=(28,28))) # 입력값을 1차원으로 펴준 뒤
    model.add(keras.layers.Dense(100, activation='relu')) # 처음 relu 활성화 함수를 이용하여 층을 넣어주고
    if a_layer : # 원한다면 하나나 몇개 더 넣을 수 있고
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax')) # 마지막은 출력층을 넣어준다.
    return model

if __name__ == '__main__':
    (train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
    train_scaled = train_input / 255.0
    train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
    
    model = model_fn()
    model.summary()
    
    ### 모델을 학습해보자
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_scaled, train_target, epochs=5, verbose=0) 
    # verbose의 기본값은 1이고 이는 학습하는 과정을 그래프의 형식으로 나타내준다.
    # verbose가 2라면 학습 과정을 막대그래프를 제외하여 그래프로 보여줌.
    # verbose가 0이라면 학습 과정을 출력하지 않는다.
    ## 학습한 반환값을 history에 넣어보았음.
    # history객체에는 훈련 측정값이 담겨있는 history 딕셔너리가 들어있다.
    print(history.history.keys()) # 'loss', 'accuracy'가 들어있음.
    # 들어있는 손실 값들과 정확도값들을 보자
    print(history.history['loss'], ' ' , history.history['accuracy']) # 각각 5개씩 들어있다. 이제 그려보자
    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3, 4, 5], history.history['loss'])
    plt.plot([1, 2, 3, 4, 5], history.history['accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('loss & accuracy')
    plt.legend()
    plt.show()
    
    
    ### 에포크를 늘렸을 때의 과대적합
    # 전에 SGD 사이킷런 모델을 공부할 때 (확률적 경사하강법에서) epoch를 늘리면 훈련 세트에 과대적합이 되어서 모델의 성능이 떨어지는 경우를 봤었음.
    # 인공 신경망 또한 전에 봤듯이 미니배치 경사 하강법을 사용하므로 일종의 경사 하강법이기 때문에 동일한 개념이 여기서도 적용된다.
    # 에포크에 따른 과대적합과 과소적합을 파악하기 위해서는 훈련 세트 뿐만이 아닌 검증 세트에 대한 점수를 같이 확인해야 한다.
    
    
    ## 전의 머신러닝에서의 확률적 경사 하강법의 과대적합 방지방식은 릿지와 라쏘를 이용하여 
    # 가중치에 절댓값 합을 더해 일부 가중치를 줄이고, 0으로 만들거나, 가중치의 제곱합을 더해서 가중치를 작게 만들어주는 방식을 사용함.
    
    ## 딥러닝에서는 딥러닝이 손실 함수를 최적화(가장 작게)하는 방식을 사용하기 때문에, 
    # epoch가 증가함에 따른 검증 세트와 훈련 세트의 손실 함수 값을 비교해보면 알 수 있을 것이다.
    model = model_fn()
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
    print(history.history.keys())
    
    plt.Figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'val'])
    plt.show()
    
    
    ### 옵티마이저를 이용하여 과대적합을 완화하는 방법
    model = model_fn()
    model.compile(optimizer='adam', loss='sparse_catrgorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
    print(history.history.keys())
    
    plt.Figure(3)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'val'])
    plt.show()
    
    
    ### 모델의 저장과 복원.
    # 모델을 만들어 놓고, 다시 훈련하고