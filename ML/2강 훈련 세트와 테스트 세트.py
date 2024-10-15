from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
    fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
    
    fish_data = [[l, r] for l, r in zip(fish_length, fish_weight)]
    fish_target = [1]*35 + [0]*14
    
    #--------------------------앞에거 볼 필요 없다.-----------
    # train_input = fish_data[:35]
    # train_target = fish_target[:35]
    #
    # test_input = fish_data[35:]
    # test_target = fish_target[35:]
    #
    # kn = KNeighborsClassifier()
    # kn = kn.fit(train_input, train_target)
    #
    # print(kn.score(test_input, test_target))
    #-------------------------------------------------------------
    
    
    # 위의 학습 데이터와 예측할 데이터를 numpy 배열로 바꾼다.
    input_arr = np.array(fish_data)
    target_arr = np.array(fish_target)
    
    
    # numpy를 이용하여 테스트 데이터를 섞은 후 다시 모델링하기.
    index = np.arange(49)  # 49개의 numpy배열을 생성함?
    print(index) # 이 index는 각 원소가 자신의 index번호를 가리키는 값을 갖는다.
    np.random.shuffle(index) # 그 index라는 numpy배열을 shuffle한다?
    print(index)
    
    # 위의 index라는 numpy배열은 그 배열 자체를 인덱스로 사용하기 위함임.
    train_input = input_arr[index[:35]]
    train_target = target_arr[index[:35]]
    
    test_input = input_arr[index[35:]]
    test_target = target_arr[index[35:]]
    # 위의 배열 슬라이싱? 방식을 이용해서 C++과는 다르게 배열의 index를 런타임에
    # 결정되는 값으로 설정하여 새로운 리스트를 생성할 수 있다.
    
    # 위에 섞인 학습 데이터와 테스트 데이터를 표현해보자.
    plt.title("shuffled data.")
    plt.scatter(train_input[:, 0], train_input[:, 1])
    plt.scatter(test_input[:, 0], test_input[:, 1])
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    

    kn = KNeighborsClassifier()
    kn = kn.fit(train_input, train_target)
    
    print(kn.score(train_input, train_target))
    print(kn.predict(test_input))
    print(test_target)
    
    
    print(kn.predict([[150, 25]]))
























    
    