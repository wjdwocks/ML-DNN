import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

### 이 전의 상황처럼 선형 회귀를 위한 데이터 설정 및 분할, 학습시키기 위한 2차원 numpy배열로의 reshpae는 기본으로 해야 함.
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
    
    train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
    # stratify=perch_weight를 하지 않는 이유는 회귀문제이지, 분류문제가 아니기 때문.
    train_input = train_input.reshape(-1, 1) # 특성이 1개이기 때문에 1차원 numpy배열로 생성되는데 이를 학습시키거나 predict할 때에는 2차원 numpy배열이 필요해서 변환해줌.
    test_input = test_input.reshape(-1, 1) # 얘도 특성이 1개이기 때문에 1차원 numpy배열로 생성되는데 이를 학습시키거나 predict할 때에는 2차원 numpy배열이 필요해서 변환해줌.
    knr = KNeighborsRegressor(n_neighbors = 3)
    knr.fit(train_input, train_target)
    
    # [[50]]으로 사용되는 이유는 2차원 numpy배열을 넘겨줘야 하기 때문.
    print('K최근접 이웃 회귀로 예측한 50cm농어의 무게 : ', knr.predict([[50]])) 
    # 원래라면 [[50], [60], [70], [80]] 으로 여러 개의 샘플을 넘겨주고, 각 특성이 여러개라면 [[50, 1], [60,2], [70,3]] 이런식임.
    
    # 이제 어떤 이웃이 선택되었는지 살펴보자.
    distances, indexes = knr.kneighbors([[50]])
    plt.figure(1)
    plt.scatter(train_input, train_target)
    plt.scatter(train_input[indexes], train_target[indexes], marker='D')
    plt.scatter(50, 1033, marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.title('Perch Weight as Length')
    plt.show()
    
    print('K최근접 이웃 회귀로 예측한 100cm 농어의 무게 : ', knr.predict([[100]]))
    plt.figure(2)
    distances2, indexes2 = knr.kneighbors([[100]])
    plt.scatter(train_input, train_target)
    plt.scatter(train_input[indexes2], train_target[indexes2], marker='D')
    plt.scatter(100, knr.predict([[100]]), marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.title('Perch Weight as Length')
    plt.show()
    
    ### 위의 상황처럼 학습시킨 데이터의 최댓값보다 더 크거나 가장 작은 경우 생기는 문제를 해결하기 위한 선형 회귀 방식.
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(train_input, train_target)
    print('선형 회귀로 예측한 50cm농어의 무게 : ', lr.predict([[50]]))
    
    # 여기서 우리는 이제 데이터를 통해서 선형 회귀가 어떻게 이루어졌는지 기울기와 y절편을 알아야 함.
    print(lr.coef_, lr.intercept_)
    # 여기서 lr.coef_는 기울기, lr.intercept_는 y절편을 나타낸다.
    # 그래서 15*lr.coef_ + lr.intercept_와 50*lr.coef_ + lr.intercept_의 두 점을 이으면 그게 선형 회귀된 모델이다.
    
    # 그러면 이제 둘 다 그려보자
    plt.figure(3)
    plt.plot([15, 50], [15*lr.coef_ + lr.intercept_, 50*lr.coef_ + lr.intercept_])
    plt.scatter(train_input, train_target)
    plt.scatter(50, lr.predict([[50]]), marker='^')
    plt.title('Linear Regression Model')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    
    ### 선형 회귀의 문제점을 찾아보자.
    print('선형 회귀 모델의 학습데이터 score vs 테스트 데이터 score : ', lr.score(train_input, train_target), lr.score(test_input, test_target))
    # 두 점수 모두 위의 K최근접 회귀 모델보다 현저히 낮다.
    # 왜그럴까?
    # 1. 점수가 낮아질 수록 차이점이 커진다.
    # 2. 아에 10cm정도보다 낮아지게 되면 음수의 무게를 갖는데 말이 안된다.
    
    ### 선형 회귀보다는 위의 데이터는 2차함수의 형태가 올바랐던 것이다.
    # 다항 회귀의 방식을 사용하여 2차식의 선형회귀를 해보자.
    train_poly = np.column_stack((train_input ** 2, train_input))
    test_poly = np.column_stack((test_input**2, test_input))
    # 각 학습, 테스트의 input들을 2차원으로 변환해줌.
    # 왜냐? 이제는 ax^2 + bx + c에 대한 target을 계산할 것이기 때문.
    lr = LinearRegression()
    lr.fit(train_poly, train_target)
    print('2차 다항식 회귀 모델에서의 50cm농어 예측값 : ', lr.predict([[50**2, 50]]))
    print('2차 다항식의 각 계수 및 y절편 : ', lr.coef_, lr.intercept_)
    
    # 그려보자.
    point = np.arange(15, 50) # 각 점에 대응하여 이을 것이기 때문에 numpy의 arange함수를 이용하여 15 ~ 49까지의 정수를 차례로 포함하는 배열을 만듦.
    plt.figure(4)
    plt.scatter(train_input, train_target)
    plt.plot(point, lr.coef_[0]*point**2 + lr.coef_[1]*point + lr.intercept_)
    plt.scatter(50, lr.predict([[50**2, 50]]), marker='D')
    plt.title('2nd Polynomial Regression Model')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    
    ### 2차 다항식 회귀로의 점수
    print('학습 데이터 점수 : ', lr.score(train_poly, train_target))
    print('테스트 데이터 점수 : ', lr.score(test_poly, test_target))
    
    