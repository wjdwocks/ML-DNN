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
    
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(fruits_2d)

print(pca.components_.shape) 
# 첫 번째 값은 50개로 변환되었다는 거고
# 두 번째 값은 원본 데이터의 특성 개수임.

# draw_fruits(pca.components_.reshape(-1, 100, 100))
# 그림으로 나타낸 주성분 50개.

print('원본 샘플들의 차원 : ',fruits_2d.shape)
fruits_pca = pca.transform(fruits_2d)
print('pca로 변환한 샘플의 차원', fruits_pca.shape)
# 특성이 10000개에서 → 50개로 줄어들었다고 생각하면 됨.

### pca로 변환한 사진샘플을 원본으로 되돌리기.
# pca의 inverse_transform 함수를 통해서 되돌릴 수 있다.
# 하지만, 이미 50개로 줄이면서 정보의 손실이 발생했기 때문에 완벽하게 되돌릴 수는 없다.
fruits_inverse = pca.inverse_transform(fruits_pca)
print('원본형태로 되돌린 데이터의 차원 : ', fruits_inverse.shape)

# 10 x 10으로 100개씩 세개의 과일을 출력하기 위해서 100 x 100으로 변환해줌
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    # draw_fruits(fruits_reconstruct[start:start+100])
    print('\n')
# 아주 잘 출력이 되는 것을 볼 수 있다.


### 설명된 분산
# pca에서는 아까 본 것 위에서 pca.components_에 주성분 50개가 들어가 있다고 했는데,
# 앞에서부터 가장 분산비율이 큰 것부터 차례로 저장되게 된다.
# 또한, 이 분산 비율들을 모두 더하면 50개의 주성분으로 표현하고 있는 총 분산 비율을 얻을 수 있다.
print(np.sum(pca.explained_variance_ratio_)) # 총 92%가 넘는 분산을 유지하는 것으로 보아, 92%가 넘는 데이터를 복원한 것임.
# plt.plot(pca.explained_variance_ratio_)
# plt.show() # 이 그림을 보면 초반 몇개의 주성분이 대부분의 분산을 담당하고 있는 것을 알 수 있다.



### 다른 알고리즘과 함께 사용하기 Logistic Regression
# 위에서 말했다시피, 특성의 개수를 주성분의 개수로 줄이기 때문에 학습하는데 걸리는 시간을 줄일 수 있다는 것을 보여줌.
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
target = np.array([0] * 100 + [1] * 100 + [2] * 100)
# print(target) # 1차원 배열로 300개의 원소가 생김.

from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target) # 모델과 전체 샘플 데이터, 정답 데이터를 넘겨줌.
print('각 폴드 당 점수의 평균 : ', np.mean(scores['test_score'])) # 각 검증에서의 점수의 평균을 계산
print('각 폴드 당 걸린 시간 평균 : ', np.mean(scores['fit_time'])) # 각 사이클에서 데이터 학습과 검증까지 걸리는 시간의 평균을 계산

### pca로 변환한 데이터로 로지스틱 회귀를 수행한 검증 결과
scores = cross_validate(lr, fruits_pca, target)
print('pca를 사용한 각 폴드 당 점수의 평균 : ', np.mean(scores['test_score'])) # 각 검증에서의 점수의 평균을 계산
print('pca를 사용한 각 폴드 당 걸린 시간 평균 : ', np.mean(scores['fit_time'])) # 각 사이클에서 데이터 학습과 검증까지 걸리는 시간의 평균을 계산
# 시간이 엄청나게 줄어든 것을 확인할 수 있다.


### pca에 n_componenets를 실수값으로 줄 수도 있다. ex) 0.5라면 위에서 본 설명된 분산(explained_valiation)이 50%를 처음 넘는값을 n_components로 자동으로 계산함.
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
print('설명된 분산이 처음 0.5를 넘는 주성분 개수 : ', pca.n_components_)
# 이 모델로 (주성분 2개) 데이터를 변환한 뒤 로지스틱 회귀에 대한 점수 확인
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

scores = cross_validate(lr, fruits_pca, target)
print('주성분이 2개인 각 폴드 당 점수의 평균 : ', np.mean(scores['test_score'])) # 각 검증에서의 점수의 평균을 계산
print('주성분이 2개인 각 폴드 당 걸린 시간 평균 : ', np.mean(scores['fit_time'])) # 각 사이클에서 데이터 학습과 검증까지 걸리는 시간의 평균을 계산

### 차원 축소된 데이터를 사용해서 k-평균 알고리즘으로 클러스터를 찾아보기
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))

### 저번 절처럼 클러스터를 이용해 분류된 과일들을 출력해보기
for label in range(0, 3):
    # draw_fruits(fruits[km.labels_ == label])
    print('\n')

### 이번에는 위에서 pca를 통해 특성을 2개로 줄였기 때문에, 그 특성을 이용해서 2차원 산점도로 표현해보자.
for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    # plt.scatter(data[:, 0], data[:,1])
# plt.legend(['apple', 'banana', 'pineapple'])
# plt.show()


print('pca로 변환된 특성 2개의 데이터 샘플', fruits_pca[:5])
print('pca로 변환된 특성 2개의 데이터 형태', fruits_pca.shape)
## 위와 같이 데이터가 생김새를 띄기에 각 label(클러스터) 를 기준으로 포함된 샘플을 data[:, 0], data[:, 1]로 점을 찍어서 그릴 수 있는 것임.




