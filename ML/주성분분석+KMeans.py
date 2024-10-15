import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

if __name__ == '__main__':
    fruits = np.load('fruits_300.npy')
    fruits_300 = fruits.reshape(-1, 100*100)
    pca = PCA(n_components=50)
    fruits_pca = pca.fit_transform(fruits_300)
    print(pca.components_)
    print(np.sum(pca.explained_variance_ratio_))
    km = KMeans(n_clusters = 3, n_init = 10)
    km.fit(fruits_pca)
    print(np.unique(km.labels_, return_counts=True))
    km = KMeans(n_clusters = 3, n_init = 20)
    km.fit(fruits_300)
    print(np.unique(km.labels_, return_counts=True))
    
    print(km.labels_)