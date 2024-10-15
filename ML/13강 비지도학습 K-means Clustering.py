import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(300, 10000)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
print("예측된 과일들.")
print(km.labels_)
print("전체 과일 개수 ", len(km.labels_))
print("각 과일 종류 당 포함된 예측된 과일 수")
print(np.unique(km.labels_, return_counts=True))

import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1):
    n= len(arr) # 배열에 포함된 전체 과일 개수. 밑을 보면 알겠지만, 배열로 값이 0인 것만 보내거나 그러고 있다.
    rows = int(np.ceil(n/10)) # 행의 개수. 천장함수를 사용하여, 91개 ~ 100개 모두 10개의 행을 가짐.
    cols = n if rows < 2 else 10 # n(배열에 포함된 과일 개수)이 11개 이상이라면 행의 개수가 2개 이상이 되어 열의 개수는 무조건 10이 되고, 그게 아니라면 n개만큼이 됨.
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False) # 저번 강의를 참고하여 그리드 형식으로 그림을 그린다.
    for i in range(rows):
        for j in range(cols):
            if i *10 + j < n:
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()
    
# draw_fruits(fruits[km.labels_ == 0])
# draw_fruits(fruits[km.labels_ == 1])
# draw_fruits(fruits[km.labels_ == 2])
print(fruits[km.labels_].shape)
print(fruits[km.labels_])

# 찾은 클러스터를 100 x 100으로 변환해주어 그림을 그려봄.
# draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)
print(km.transform(fruits_2d[[100]]))
print(km.predict(fruits_2d[100:101]))
# draw_fruits(fruits[100:101])
print(km.n_iter_)
print(km.labels_.shape)
print(km.cluster_centers_.shape)


### 최적의 K 찾기 (엘보우 메소드)
inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters = k, random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
    
plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
print(inertia)
