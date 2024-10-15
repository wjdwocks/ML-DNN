from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
    fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
    
    # 아레의 코드는 fish_length, fish_weight의 각 index를 하나의 열로 하여 쌓아준다는 의미.
    fish_data = np.column_stack((fish_length, fish_weight))
    print(fish_data[:5])
    
    # 아레의 코드는 두 numpy 배열을 하나의 numpy배열로 합치는 함수이다.
    # tuple의 형태로 이어붙이고 싶은 numpy배열들을 넣어준다.
    fish_target = np.concatenate((np.ones(35), np.zeros(14)))
    print(fish_target)
    
    
    ### scikitlearn을 통해 전체 데이터 중 훈련 데이터와 테스트 데이터를 나누는 방법.
    # 위에 from sklearn.model_selection import train_test_split을 추가하고
    # 아레의 함수를 통해서 분류받는다.
    train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
    # 여기에서 random_state는 필수가 아님. 하지만 앞에 데이터를 하나씩 받는 순서는 익혀두어야 한다.
    # 이 함수는 기본적으로 25%의 데이터를 테스트 세트로 떼어내고 나머지를 학습 데이터로 둔다.
    print(train_input.shape, test_input.shape , " : 이것을 보면 3:1 비율로 학습 : 테스트 가 된 것을 볼 수 있다.")
    print(train_target.shape, test_target.shape, " : 이것을 보면 3:1 비율로 잘 섞임.")
    
    # 하지만 내부로는 랜덤으로 섞였기 때문에 운이 없다면 샘플링 편향이 발생하게 될 수도 있다.
    # 예를 들어 이 경우에도 원래는 물고기가 2.5:1의 비율로 있지만 테스트 비율은 3:1정도 이다.
    print(test_target, " : 하지만 이것을 보면 비율이 마음에 들지 않는다. 샘플링 편향이 발생할 수도 있다." )
    
    # 그렇기 때문에 위에 train_test_split함수의 매개변수로 stratify에 타깃 데이터를 전달해주면 클래스 비율에 맞게 데이터를 나누어준다.
    train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
    print(test_target, "비율이 2.5:1에 더 가까워졌다.")
    
    # 이렇게 다시 만들어진 데이터들로 다시 KN모델에 넣어봄.
    kn = KNeighborsClassifier()
    kn.fit(train_input, train_target)
    print(kn.score(test_input, test_target))
    
    print("25, 150인 도미를 1로 예측하는가? : ", kn.predict([[25, 150]]), " 아니었다..")
    
    # 만약 kneighbor를 사용할 때 최근접 점에 어떤 것이 선택되는지를 확인하고 싶다면?
    distances, indexes = kn.kneighbors([[25,150]]) # 이 코드는 (25, 150)이라는 점에 가장 가까운 점부터 거리와 index 들을 2차원 numpy배열의 형태로 각각 반환해줌.
    # 이는 kneighbors의 n_neighbors가 기본값인 5로 되어있기 때문에 distances와 indexes의 원소는 각각 5개씩일 것이다.
    print(distances, " ", indexes)
    
    
    # 이러한 큰 도미를 산점도로 나타내서 왜 이런 결과가 나타나는지 확인해봄.
    # plt.title('fish data')
    # plt.scatter(train_input[:, 0], train_input[:, 1]) # 학습한 데이터의 산점도
    # plt.scatter(25, 150, marker='^') # 수상한 도미의 점.
    
    # 위에서 얻어낸 최근접 점들의 index들을 scatter를 표시하는 데 이용할 수 있다.
    # plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
    # plt.xlim(0, 1000)
    # plt.xlabel('length')
    # plt.ylabel('weight')
    # plt.show
    
    ### 위의 정보를 보면 우리의 눈으로 보이는 최근접 점과 실제 거리는 다르다.
    # 왜냐하면 x축은 0~40으로 되어있고 y축은 0~1000으로 눈으로 보이는 것보다 y축에의한 값이 지배적이다.
    # 기준을 맞추어주어야 함. (스케일을 지정해줌.)
    # 가장 널리 사용되던 방식은 각 특성(길이, 무게)를 표준점수로 변환하여 ((특성-평균) / 표준편차) 각 특성에 대한 점수를 따로 합치는 방식이다.
    # 여기에서는 numpy의 mean함수로 평균을 구하고, std함수로 표준 편차를 구한 다음 직접 계산을 해보도록 하자.
    mean = np.mean(train_input, axis=0)
    std = np.std(train_input, axis=0)
    # axis는 축을 말하고, 0을 부여하면 각 특성별로(열별로) 평균과 표준편차를 계산해주게 된다.
    print(mean, std)
    # mean = [ 27.29722222 454.09722222]
    # std = [9.98244253 323.29893931]
    # 길이의 평균, 무게의 평균
    # 길이의 표준편차, 무게의 표준편차
    train_scaled = (train_input - mean)
    # 각 train_input의 (길이, 무게)를 mean에서 뺀 뒤 반환된다. 이 값도 2차원 numpy배열.
    # [ -15.09722222 -441.89722222] 이런식이 됨.
    train_scaled = train_scaled / std
    
    plt.scatter(train_scaled[:, 0], train_scaled[:, 1]) # 각 생선별로, 0(길이) 1(무게)
    plt.xlabel('length')
    plt.ylabel('ylabel')
    
    
    # 그런데 여기서 수상한 생선(25, 150)을 표시하려고 한다면 테스트 데이터의 평균과 표준편차로 표준점수를 계산해야하는가 아니면 
    # 학습 데이터의 평균과 표준편차로 표준점수를 계산해야 하는가가 문제가 될 수 있다.
    # 이것은 당연히 학습 한 데이터의 표준편차와 평균으로 계산해야 한다. (그것이 이 알고리즘의 척도가 될 테니까)
    new = ([25, 150] - mean) / std
    print(new)
    plt.scatter(new[0], new[1], marker='^')
    
    plt.show
    
    ### 이제 이렇게 변환된 데이터로 다시 모델을 훈련시킨다.
    kn.fit(train_scaled, train_target)
    test_scaled = (test_input - mean) / std # 테스트 할 데이터들도 표준점수로 변환해준다.
    print(kn.score(test_scaled, test_target))
    
    print(kn.predict([new]))
    
    # 다시 kn.neighbors함수로 어떤 것들이 최근접 점으로 선택되었는지 확인해보자
    distances, indexes = kn.kneighbors([new])
    plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
    plt.scatter(new[0], new[1], marker='^')
    plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.title('Strange Fish')
    plt.show
    
    