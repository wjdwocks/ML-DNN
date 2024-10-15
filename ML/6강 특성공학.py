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
    print(perch_full)
    print('데이터의 개수 : ', perch_weight.size, '개')
    
    train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
    poly = PolynomialFeatures()
    poly.fit([[2, 3]]) # 2와 3으로 이루어진 특성을 학습시킴.
    print(poly.transform([[2,3]])) # [[1 2 3 4 6 9]]가 나오게 되는데, 이는 2와 3중 중복을 포함하여 2번 이하로 포함하는 경우를 모두 나타냄.
    # 그런데 1은 각 특성을 제곱하고, 곱한 특성이 아닌 자동으로 절편이 1인 특성과 곱해진다고 여겨진다.
    # 그래서 PolynomialFeatures()클래스 객체를 생성할 때 include_bias=False를 달아주면 자동으로 추가를 안함.
    poly = PolynomialFeatures(include_bias=False)
    poly.fit([[2, 3]])
    print(poly.transform([[2,3]]))
    ################### 이제 위의 train_input에 대해 전처리(fit, transform)을 해보자.
    poly.fit(train_input)
    train_poly = poly.transform(train_input)
    print(train_poly.shape) # (42, 9) 가 나옴.
    # 즉, 위의 56개의 데이터 중 42개가 랜덤으로 선택되고, 각 데이터는 9개의 특성을 갖고 있다.
    # 그렇다면 각 특성은 무엇인가?
    # 아레의 함수를 통해서 어떤 것들인지 알 수 있다.
    print(poly.get_feature_names_out())
    # 각각의 1차식과 3개 중 두개를 곱한 값(3C2), 각각의 제곱이 선택되어 있다.
    
    test_poly = poly.transform(test_input) # 테스트 값도 위와 같이 PolynomialFeatures객체를 이용하여 전처리를 해준다.
    
    ### 위에서 PolynomialFeatures객체를 이용하여 전처리한 train데이터와 test데이터를 이용해서 선형회귀 모델로 학습해보자.
    lr = LinearRegression()
    lr.fit(train_poly, train_target)
    print('학습 데이터에 대한 점수 : ', lr.score(train_poly, train_target))
    print('테스트 데이터에 대한 점수 : ', lr.score(test_poly, test_target))
    print('학습 데이터 점수 > 테스트 데이터 점수 이므로 과소적합은 더이상 나타나지 않음.')
    
    
    # 여기까지는 PolynomialFeatures() 클래스에 매개변수를 넘겨주지 않아서 기본적으로 2차식으로 전처리된 특성을 사용한 예시이다.
    # 아레는 5제곱까지의 특성을 만들어서 해보자.
    ### PolynomialFeatures()에 degree=5를 넣어줌.
    poly = PolynomialFeatures(degree=5, include_bias=False)
    poly.fit(train_input)
    train_poly = poly.transform(train_input)
    # poly.fit(test_input) 이거는 쓰면 안됨. fit은 학습이고, 학습 데이터로 학습한 대로 test데이터도 변환해주어야 한다.
    test_poly = poly.transform(test_input)
    print(train_poly.shape) # (42, 55)가 나옴 특성이 무려 55개라는 소리.
    
    ### 이제 위에서 한 5제곱의 다항식으로 전처리된 데이터로 학습을 해보자.
    lr = LinearRegression()
    lr.fit(train_poly, train_target)
    print('학습된 데이터에 대한 점수 : ', lr.score(train_poly, train_target), '테스트 데이터에 대한 점수 : ', lr.score(test_poly, test_target))
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
    from sklearn.linear_model import Ridge
    ridge = Ridge()
    ridge.fit(train_scaled, train_target)
    print('릿지로 학습된 학습데이터의 점수 : ', ridge.score(train_scaled, train_target))
    print('릿지로 학습된 테스트 데이터에 점수 : ', ridge.score(test_scaled, test_target))
    
    
    
    
    
    
    
    
    
    
    
    