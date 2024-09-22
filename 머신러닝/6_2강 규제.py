import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    df = pd.read_csv("https://raw.githubusercontent.com/rickiepark/hg-mldl/master/perch_full.csv")
    perch_full = df.to_numpy()
    perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
    print('데이터의 개수 : ', perch_weight.size, '개')
    
    train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
 
    poly = PolynomialFeatures(degree=5, include_bias=False)
    poly.fit(train_input)
    train_poly = poly.transform(train_input)
    # poly.fit(test_input) 이거는 쓰면 안됨. fit은 학습이고, 학습 데이터로 학습한 대로 test데이터도 변환해주어야 한다.
    test_poly = poly.transform(test_input)
    
    ### 이제 위에서 한 5제곱의 다항식으로 전처리된 데이터로 학습을 해보자.
    lr = LinearRegression()
    lr.fit(train_poly, train_target)
    print('\n학습된 데이터에 대한 점수 : ', lr.score(train_poly, train_target), '\n테스트 데이터에 대한 점수 : ', lr.score(test_poly, test_target))
    # 테스트 점수가 -144점이 나옴.
    # 이것이 바로 과대적합이다.
    
    ### 과대 적합을 줄일 수 있는 방법 : 규제(Regularization)
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    ss.fit(train_poly) # 여기서의 train_poly는 5제곱된 걔네가 맞다.
    train_scaled = ss.transform(train_poly)
    test_scaled = ss.transform(test_poly)
    # 위에서 StandardScaler는 각 특성을 표준 점수로 바꾸어주는 변환기의 역할을 수행한다.
    # 그렇기 때문에 우리가 전에 분류문제를 해결할 때 처럼 학습 데이터를 통해서 fit된 그 표준점수 기준으로 test데이터도 transform(변환)해주어야 함.
    
    ## 릿지와 라쏘 : 선형 회귀 모델에 규제를 추가한 모델을 부르는 말.
    # 두 모델은 규제를 가하는 방식이 다르다. 릿지는 계수를 제곱한 값을 기준으로 규제를 적용.
    # 라쏘는 계수의 절댓값을 기준으로 규제를 적용한다.
    # 일반적으로 릿지를 조금 더 선호하는데, 두 알고리즘 모두 계수의 크기를 줄이는 역할을 하지만, 라쏘는 아에 0으로도 만들어버린다.
    
    ### 릿지 회귀 : 사이킷런 모델의 장점은 훈련하고, 사용하는 방식이 항상 같다는 점이다.
    # 그래서 그냥 클래스 객체를 생성하고, 훈련 데이터를 학습시킨 뒤 점수를 도출하면 된다.
    from sklearn.linear_model import Ridge
    ridge = Ridge()
    ridge.fit(train_scaled, train_target)
    print('\n릿지로 학습된 학습데이터의 점수 : ', ridge.score(train_scaled, train_target))
    print('릿지로 학습된 테스트 데이터에 점수 : ', ridge.score(test_scaled, test_target))
    
    
    ### 릿지나 라쏘 회귀를 할 때 사용할 수 있는 alpha 매개변수.
    # alpha는 각 특성의 계수를 얼마나 줄일 지를 조절하는 매개변수이다.
    # alpha가 작다면 줄이는 계수가 작고, alpha가 크다면 크게 줄이게 된다.
    import matplotlib.pyplot as plt
    train_score = [] # 각 alpha값 당 학습 데이터 점수를 저장하기 위한 리스트 생성
    test_score = [] # 각 alpha값 당 테스트 점수를 저장하기 위한 리스트 생성
    
    alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
    for alpha in alpha_list:
        ridge = Ridge(alpha)
        ridge.fit(train_scaled, train_target)
        train_score.append(ridge.score(train_scaled, train_target))
        test_score.append(ridge.score(test_scaled, test_target))
    
    # 만약 위의 alpha_list의 값들을 그대로 plt에 그리면 0.001 ~ 0.1간의 간격과 0.01~100의 간격이 너무 차이가 심하므로 로그 함수에 넣어서 x값을 변환해준다.
    # log10에 넣게 되면 -3, -2, -1, 0, 1, 2 가 나와서 각 x축의 간격이 동일하게 될 것이다.
    plt.scatter(np.log10(alpha_list), train_score)
    plt.scatter(np.log10(alpha_list), test_score)
    plt.xlabel('alpha values')
    plt.ylabel('train/test scores')
    plt.show()
    # -1에서 가장 두 점의 거리가 가깝다. ← 이 뜻은 alpha가 0.1일 때 가 가장 적절하다는 의미이다.
    # 최종적으로 alpha = 0.1로 두고 학습을 하면
    ridge = Ridge(0.1) # alpha=0.1로 하면 좋지만, 그냥 0.1을 넘기면 alpha에 자동으로 넘어감. (가장 많이 쓰이는 parameter라서 그런듯.)
    ridge.fit(train_scaled, train_target)
    print('\nalpha = 0.1일 때의 \n학습 데이터 점수 : ',ridge.score(train_scaled, train_target), '\n테스트 데이터의 점수 : ', ridge.score(test_scaled, test_target))
    
    ### 라쏘 회귀.
    # 라쏘 회귀의 학습 방법과 alpha값을 찾는 방법은 위의 릿지와 그냥 똑같다. 한번 작성해보자.
    from sklearn.linear_model import Lasso
    lasso = Lasso()
    lasso.fit(train_scaled, train_target)
    print('\n라쏘로 학습한 \n학습 데이터 점수 : ', lasso.score(train_scaled, train_target), '\n테스트 데이터 점수 : ', lasso.score(test_scaled, test_target))
    
    # 라쏘의 최적의 alpha값을 찾기.
    lasso_train_score = []
    lasso_test_score = []
    
    for alpha in alpha_list:
        lasso = Lasso(alpha=alpha)
        lasso.fit(train_scaled, train_target)
        lasso_train_score.append(lasso.score(train_scaled, train_target))
        lasso_test_score.append(lasso.score(test_scaled, test_target))
    
    plt.figure(2)
    plt.plot(np.log10(alpha_list), lasso_train_score)
    plt.plot(np.log10(alpha_list), lasso_test_score)
    plt.xlabel('alpha_list')
    plt.ylabel('$R^2$ values')
    plt.show()
    
    # alpha가 10일 때 가장 잘 학습되어있다고 볼 수 있다.
    print('\nalpha=10일 때 라쏘로 학습한 \n학습 데이터 점수 : ', lasso_train_score[4], '\n테스트 데이터 점수 : ', lasso_test_score[4])
    
    ## 라쏘 모델에서 0이 된 계수를 알아보자.
    lasso = Lasso(alpha=10)
    lasso.fit(train_scaled, train_target)
    print('계수가 0이 되어버린 특성의 개수 : ', np.sum(lasso.coef_ == 0))
    # alpha가 10일 때 0인 계수가 40개나 됬다. 유용한 특성을 골라내는 데에 사용할 수 있을 것이다.
    