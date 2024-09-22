from tensorflow import keras
from sklearn.model_selection import train_test_split

def model_fn(a_layer = None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer :
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

if __name__ == '__main__' :
    ### 데이터 불러오기
    (train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
    train_scaled = train_input/255.0
    train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
    
    # model = model_fn(keras.layers.Dense(30, activation='sigmoid'))
    model = model_fn(keras.layers.Dropout(0.3))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_scaled, train_target, epochs=10, verbose=0, validation_data =(val_scaled, val_target))
    
    ### 모델 저장과 복원
    model.save_weights('model.weights.h5') # 모델의 가중치만 저장, 같은 구조로 다시 만든 모델에 이 가중치를 load해주면 전과 같은 결과를 냄.
    model.save('model.whole.h5') # 모델의 모든 가중치와 구조, 컴파일러 구조, 옵티마이저 등등 모든 것을 그대로 불러온다.
    
    ## 훈련을 하지 않은 새로운 모델을 만들고, model.weights.h5를 로드해서 사용해보기.
    model = model_fn(keras.layers.Dropout(0.3)) # 이 모델은 위의 model.save_Weights('model.weights.h5')에서의 model과 완전히 같은 구조를 가져야 한다.
    model.load_weights('model.weights.h5')
    
    # 위의 load_weights로 가중치를 가져온 모델의 검증 정확도를 확인해보자.
    # predict 메서드는 사이킷런과 달리 샘플마다 10개의 클래스에 대한 확률을 반환해서 가장 높은 확률을 그 샘플의 클래스로 예측한다.
    # 검증 데이터 세트가 12000개였기 때문에 위의 모델로 predict(val_scaled)를 하게 되면 (12000, 10)의 배열이 생성된다.
        
    import numpy as np
    val_labels=np.argmax(model.predict(val_scaled), axis=-1) # model.predict(val_scaled)를 하면 (12000, 10)의 배열이 생성되고, 
    # numpy의 argmax()함수를 axis = -1로 넘겨주어서 각 12000개의 샘플이 각 행에 세로로 위치한다면, 가로 10개의 클래스 중 가장 큰 값을 선택하게 된다.
    # axis = 0이라면 세로 방향으로 각 열에 포함되는 원소들의 최댓값을 고르게 된 값들로 새로운 배열이 형성되고
    # axis = 1이라면 가로 방향으로 각 행에 포함되는 원소들의 최댓값을 고르게 된 값들로 새로운 배열을 만들게 된다.
    print(np.mean(val_labels == val_target)) # 그렇게 각 12000개의 값을 가진 배열과 (12000개의 샘플의 정답이 들어있는 배열을 각 index별로 비교하여
    # 같으면 true(1), 다르면 false(0)으로 하여 평균을 낸다. 이렇게 검증 세트에 대한 정답률을 구한다.

    ### 모델 전체를 저장하고 비교해보자
    model = keras.models.load_model('model.whole.h5')
    model.evaluate(val_scaled, val_target)
    
    
    ### 콜백
    model = model_fn(keras.layers.Dropout(0.3))
    model.compile(optimizer='adam', loss ='sparse_categorical_crossentropy'
                  , metrics=['accuracy'])
    checkpoint_cb = keras.callbacks.ModelCheckpoint('best_model.keras', 
                                                    save_best_only=True)

    model.fit(train_scaled, train_target, epochs=20,
              validation_data=(val_scaled, val_target),
              callbacks=[checkpoint_cb])
    
    model = keras.models.load_model('best_model.keras')
    model.evaluate(val_scaled, val_target)
