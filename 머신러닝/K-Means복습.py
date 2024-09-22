import numpy as np
import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:
                axs[i, j].imshow(arr[i*10+j], cmap='gray')
            axs[i, j].axis('off')
    plt.show()
        
if __name__ == '__main__':

    fruits = np.load('fruits_300.npy')

    fruits_2d = fruits.reshape(-1, 10000) # 비지도학습이므로 300개의 데이터(과일)과 각각 10000개의 특성을 갖도록 재구성
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3) # 중심이 될 클러스터의 개수를 정해줌.
    km.fit(fruits_2d) # 비지도 학습이기 때문에 (샘플, 특성들)의 2차원 numpy배열만 받는듯.
    print('각 과일에대한 예측 : ',km.labels_) # 이 km.labels는 넘겨진 데이터(300개)가 각각 어떤 클래스로 예측이 되었는지 들어있다.
    print('전체 과일 수 : ', km.labels_.shape)
    print('각 클러스터(과일종류)에 속한 개수 : ', np.unique(km.labels_, return_counts=True))

    draw_fruits(fruits[km.labels_==0])
    draw_fruits(fruits[km.labels_==1])
    draw_fruits(fruits[km.labels_==2])

    print(km.transform(fruits_2d[100:102]))
    
    inertia = []
    for k in range(2, 8):
        km = KMeans(n_clusters = k)
        km.fit(fruits_2d)
        inertia.append(km.inertia_)
        
    plt.plot(range(2, 8), inertia)
    plt.xlabel('k')
    plt.ylabel('inertia')
    plt.show()