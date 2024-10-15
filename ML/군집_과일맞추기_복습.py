import numpy as np
fruits = np.load('fruits_300.npy')

print(fruits.shape)
apple = fruits[0:100].reshape(-1, 10000)
pineapple = fruits[100:200].reshape(-1, 10000)
banana = fruits[200:300].reshape(-1, 10000)

# 200번째 과일이 무엇인지 imshow함수로 출력해보자.
import matplotlib.pyplot as plt
# plt.imshow(fruits[200], cmap='gray')
# plt.show()

# 히스토그램으로 한 사진 당 모든 픽셀값의 평균을 히스토그램으로 나타내서 학습의 척도로 사용할 수 있는가?    
# plt.hist(np.mean(apple, axis=1)) # axis = 1로 pineapple은 크기가 [100][10000]일텐데 뒤의 10000크기를 평균내게 된다.
# plt.hist(np.mean(pineapple, axis=1))
# plt.hist(np.mean(banana, axis=1))
# plt.show()

# 각 픽셀(10000개의) 당 평균을 내서 평균사진이라는 것을 만든 뒤, 비교할 사진의 각 픽셀 당 절댓값 차이와 비교하여 학습.
# plt.bar(range(10000), np.mean(apple, axis=0))
# plt.bar(range(10000), np.mean(pineapple, axis=0))
# plt.bar(range(10000), np.mean(banana, axis=0))
# plt.legend(['apple', 'pineapple', 'banana'])
# plt.show()
# 하지만 위에처럼 그리면 한 화면에 10000개의 픽셀들이 모두 나옴. 이러면 보기엔 나쁘지 않은데, 각 화면으로 봐보자.
# fig, axs = plt.subplots(1, 3, figsize=(12, 4))
# axs[0].bar(range(10000), np.mean(apple, axis=0))
# axs[1].bar(range(10000), np.mean(pineapple, axis=0))
# axs[2].bar(range(10000), np.mean(banana, axis=0))
# plt.show()

# 이젠 이걸 이용해서 분류하는 알고리즘을 작성해보자.
apple_mean = np.mean(apple, axis=0)
pine_mean = np.mean(pineapple, axis=0)
banana_mean = np.mean(banana, axis=0)
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(apple_mean.reshape(100, 100), cmap='gray')
axs[1].imshow(pine_mean.reshape(100, 100), cmap='gray')
axs[2].imshow(banana_mean.reshape(100, 100), cmap='gray')
plt.show()

# 데이터를 fruits와 맞춰주기 위해 (100, 100)의 크기로 다시 만들어줌.
apple_mean = apple_mean.reshape(100, 100)
pine_mean = pine_mean.reshape(100, 100)
banana_mean = banana_mean.reshape(100, 100)

abs_diff = np.abs(fruits-apple_mean) # 각 픽셀끼리 모든 과일들과 사과의 평균을 뺄셈함.
print(abs_diff.shape)
abs_mean = np.mean(abs_diff, axis=(1,2))

apple_index = np.argsort(abs_mean)[:100]
print(apple_index)
print(abs_mean)

fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[10*i + j]], cmap='gray')
plt.show()

abs_mean = np.array([50, 20, 30, 10, 40])
sorted_index = np.argsort(abs_mean)
print(sorted_index) # [3 1 2 4 0]
print(abs_mean[sorted_index[:5]])