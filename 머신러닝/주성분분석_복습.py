import numpy as np
import matplotlib.pyplot as plt

def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10 # n이 한자리수면 n, 아니면 10.
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(rows):
        for j in range(cols):
            if i * 10 + j < n:
                axs[i, j].imshow(arr[i*10 + j], cmap='gray')
            axs[i, j].axis('off')
    plt.show()

from sklearn.decomposition import PCA

if __name__ == '__main__':
    fruits = np.load('fruits_300.npy')
    fruits_2d = fruits.reshape(-1, 10000)
    pca = PCA(n_components=50)
    pca.fit(fruits_2d)
    
    print(pca.components_.shape) # 50개로 변환된 특성, 원본의 특성 개수
    print(pca.components_[0])
    
    draw_fruits(pca.components_.reshape(-1, 100, 100))
    
    fruits_pca = pca.transform(fruits_2d)
    print('pca로 변환한 샘플의 차원 : ', fruits_pca.shape)
    
    fruits_reverse_pca = pca.inverse_transform(fruits_pca)
    print('원본의 복원률 분산 : ', pca.explained_variance_ratio_())
    
    