import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

if __name__ == '__main__':
    perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
    perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
    
    plt.scatter(perch_length, perch_weight)
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    
    # 앞에서 배웠던 것처럼 전체 데이터와 답이 있어야하는데
    # 여기서는 길이와 무게의 상관관계를 보기 위해 src값이 길이 하나이고, 답이 weight값인 것이다.
    train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
    # 위와 같이 데이터가 1차원 배열일 때 문제가 발생할 수 있다. 왜냐하면 sklearn은 2차원 numpy배열을 받아야 하기 때문이다.
    # 그렇기 때문에 train_input과 test_input을 2차원 numpy배열로 reshape해주어야 한다.
    
    
    train_input = train_input.reshape(-1, 1)
    test_input = test_input.reshape(-1, 1)
    # 위와 같이 각 열은 1개(길이의 특성)으로 되고, 각 행은 각 농어들(샘플)이므로 2차원 배열이 된다.
    
    knr = KNeighborsRegressor()
    knr.fit(train_input, train_target)
    
    # Classifier모델에서는 전체 테스트 데이터 중 맞춘 개수가 score이다.
    # Regressor 모델에서의 score(R)는 R^2 = 1 - {∑(타킷-예측)^2 / ∑(타깃-평균)^2} 이다.
    print("Regressor 모델에서의 score는 1 - {∑(타킷-예측)^2 / ∑(타깃-평균)^2} 이다.", knr.score(test_input, test_target))
    
    test_prediction = knr.predict(test_input)
    print(test_input, "||" , test_prediction)
    mae = mean_absolute_error(test_target, test_prediction)
    print("평균 절댓값 오차 : ", mae)
    
    ### 과대적합과 과소적합.
    print("과대 적합과 과소적합. 이미 학습시킨 데이터의 score가 test케이스의 데이터의 score보다 낮게 나옴.", knr.score(train_input, train_target), "vs" , knr.score(test_input, test_target))
    
    # knr에서 n_neighbors의 수를 줄이면 과대적합, 수를 늘리면 과소적합이 됨.
    # n_neighbors가 작다면 (1에 가깝다면) 예측값이 실제값과 완전히 똑같아야 된다고 생각하게 됨.
    # n_neighbors가 크다면 너무 멀리있는 값까지도 회귀할 때 값으로 사용하기 때문에 의미가 옅어짐.
    knr.n_neighbors = 3
    knr.fit(train_input, train_target)
    print(knr.score(train_input, train_target))
    print(knr.score(test_input, test_target))

    
    