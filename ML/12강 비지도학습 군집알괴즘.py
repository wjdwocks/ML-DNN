import numpy as np
import matplotlib.pyplot as plt

# 100 x 100px로 되어있는 과일 그림 샘플 300개가 fruits에 저장된다.
fruits = np.load('C:/Users/user/Downloads/fruits_300.npy')

print(fruits.shape) # 그래서 300, 100, 100이라는 형태임.

print(fruits[0,0,:]) # fruits는 100x100짜리가 300개 있는 것임.
# 그래서 위와 같이 출력을 하면 첫번 째 그림의 0번 행에 있는 100개의 열을 출력해줌.

### px형식으로 저장된 numpy배열을 그리는 imshow함수.
plt.imshow(fruits[0], cmap='gray') # 첫 번째 그림에 100 x 100px의 값이 저장되어있으므로 그거를 그려줌.
plt.show() 
# cmap='gray'는 흑백 사진이라는 것을 의미함.
# 칼라 사진이면 각 px당 (r, g, b)가 있었겠지?
plt.imshow(fruits[1], cmap='gray_r')

### matplotlib에서 여러 장의 그림을 동시에 보는 방법.
fig, axs = plt.subplots(1, 2) # 하나의 도화지에 1행 2열의 그림을 그림.
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()
# 1행 2열로 그림을 나열 할 것이기 때문에 1행의 [0]번 위치, [1]번 위치에 저 그림을 둔다고 생각하면 됨.


### 우리는 처음으로 그림을 구분하고, 분류하기 위해서 각 사진마다의 전체 픽셀의 값을 평균내서 하려고 한다.
# 일단 사과, 바나나, 파인애플들의 사진들을 분리한다.
apple = fruits[0:100].reshape(100, 100*100) # fruits[0]~fruits[99]이 사과에 해당하니 걔네만 빼고, 뒤의 [100][100]은 한차원이 10000을 가지도록 합친다.
pineapple = fruits[100:200].reshape(-1, 100*100) # reshape의 -1은 앞의 차원은 알아서 하라는 거임. 즉, reshape(100, 10000)임. 각 샘플이 100개씩 있기 때문에.
banana = fruits[200:300].reshape(-1, 100*100)
print(apple.shape)

### 이제, 각 사진마다 픽셀값들의 평균값을 계산해 보자.
# numpy의 mean함수를 이용함.
# mean함수의 axis는 축을 의미하는데 axis=0이면 행 방향으로 ↓ 더해서 평균을 내고
# axis=1이면 열 방향으로 → 더해서 평균을 내게 된다.
# 여기서는 우리가 각 과일마다 픽셀값을 열로 나타내었기 때문에 axis=1로 두면 됨.
# 1번 과일 : (픽셀들)
# 2번 과일 : (픽셀들) 

print(apple.mean(axis=1)) # apple 배열에 들어있는 각 사진들에 대해서 평균을 내서 다시 리스트로 나옴.
### 히스토그램으로 각 평균 값들을 그려보기
plt.figure(3)
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()



### 다시, 각 사진마다의 전체 픽셀 평균이 아닌, 각 픽셀마다의 평균값을 구해보자.
# fig, axs = plt.subplots(1, 3, figsize=(15,5))
# axs[0].bar(range(10000), np.mean(apple, axis=0))
# axs[1].bar(range(10000), np.mean(pineapple, axis=0))
# axs[2].bar(range(10000), np.mean(banana, axis=0))
# plt.show()

### 그러면 우리는 이 각 과일의 각 픽셀마다의 평균값으로 만든 그림과 비슷한 그림이면 그 그림으로 판단할 수 있을 것이다.
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')


### 전체 과일들 중 apple_mean과 각 픽셀들의 절댓값 오차의 평균이 가장 작은 100개를 고르면 모두 다 사과일까?
abs_diff = np.abs(fruits-apple_mean) # 각 n번째 과일들 당 [100][100]의 픽셀들 별로 뺄셈을 수행함.
abs_mean = np.mean(abs_diff, axis=(1,2)) # 300, 100, 100의 샘플에서 axis=(1,2)로 줌으로써, 100x100 픽셀, 10000개의 데이터에 대해서 평균을 낸다.
print(abs_mean.shape) # 그래서 300, 로 나오는거임. 각 300개의 과일에 대해서 apple_mean과의 절댓값 오차의 평균만 가지고 있기 때문에.

### 이제 오차가 가장 적은 100개의 과일만 추출해서 그려보자.
apple_index = np.argsort(abs_mean)[:100] # argsort는 크기 순서대로 배열을 정렬하는 것이다. 그리고 그것의 [:100]앞의 100개를 선택
fig, axs = plt.subplots(10, 10, figsize=(10, 10)) # axs[i, j]번째 순서에 그림을 그릴거임.
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[10*i+j]], cmap='gray_r') # apple_index[i*10 + j] 를 해주어야 아까 정렬한 그 순서대로 출력함. 따로 변수를 안만드려고 그러는듯.
        # axs[i, j].imshow(fruits[apple_index[wocks]], cmap='gray_r') 물론 이렇게 해도 돌아가는건 똑같다.
        # wocks += 1
        axs[i, j].axis('off') # 축을 안나누고 그릴거임.
    plt.show()








