### 6개의 특성을 가진 물고기 데이터를 pandas를 통해 가져오자. (csv파일)
import pandas as pd
fish = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fish.csv')
print(pd.unique(fish['Species'])) # 각 물고기들은 Species가 7개로 정해져 있기 때문에 unique함수를 통해서 어떤 종이 있는지 확인해볼 수 있다.

### 가져온 물고기 데이터를 0열의 종의 분류, 1~5열의 각 특성으로 나누자.
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy() # 얘는 입력 데이터가 될 것이므로 2차원 numpy배열로 생성.
print(fish_input[:5]) # fish_input의 앞의 5개만 출력해보자.

fish_target = fish['Species'].to_numpy() # 얘는 입력 2차원 numpy배열의 정답값이 될 것이므로 1차원임.
print(fish_target[:5])

### train_test_split함수를 통해 학습 데이터와 테스트 데이터로 나누자.
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

### StandardScaler클래스를 이용하여 훈련 데이터와 테스트 데이터를 표준화한다.
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

### k-Clustering Classifier 모델을 이용하여 (각 특성들을 통해) 물고기의 종류를 학습시켜보자. 
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors = 3)
kn.fit(train_scaled, train_target)
print('학습 데이터 점수 : ', kn.score(train_scaled, train_target), '\n테스트 데이터 점수 : ', kn.score(test_scaled, test_target))

#-----------------------------------------------------------------------------------------#
print('\ntarget에 있는 종류들 : ', kn.classes_)
print('\n테스트 데이터의 앞의 5개에 대한 예측 값 : ', kn.predict(test_scaled[:5]))
print('\n테스트 데이터에 대한 앞의 5개의 실제 값 : ', test_target[:5])

### 이제 우리는 처음부터 각 생선이 될 확률이 알고 싶었으므로 그 확률을 출력하는 방법을 알아보아야 함.
import numpy as np
proba = kn.predict_proba(test_scaled)
print(kn.classes_)
print(np.round(proba, decimals=4))

### 그렇다면 3번째 샘플을 예측할 때 선택된 점들을 확인해보자.
distances, indexes = kn.kneighbors([test_scaled[3]])
print(train_target[indexes])

##### 로지스틱 회귀
# 로지스틱 회귀는 이름만 회귀이지 분류 알고리즘이다.
# 로지스틱 회귀는 선형 방정식을 학습하는데,  z = a*(Weight) + b*(Length) + c*(Diagonal) + d*(Height) + e*(Width) + f 이런꼴이다.
# z의 값이 확률이 될 것이라서 0 ~ 1사이의 값을 가져야 한다.
# 여기서 사용하는게 시그모이드 함수이다.
# 시그모이드 함수는 1 / (1+e^(-z)) 이다.
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z)) # z의 개수가 많으니까 각 z당 리스트로 생길것임.
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

### 넘파이 배열의 boolean indexing 방법
# 넘파이 배열을 True, False로 원소를 넣어주면 print()하면 True인 얘만 출력됨.
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr)
print(char_arr[[True, False, True, False, False]])
# 여기서 왜 char_arr[]안에 배열이 들어갔냐면 각 원소에 대응하는 값을 리스트의 형태로 전달해주기 위함임.
# 위와 같이 훈련 세트에서 도미와(Bream)과 빙어(Smelt)만 골라내는 방법.
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt') # 여기에서 이제 target값이 Bream이나 Smelt라면 True, 아니라면 False가 들어가게 된다.
train_bream_smelt = train_scaled[bream_smelt_indexes] # bream이나 smelt물고기에 대한 정보만 들어있음.
target_bream_smelt = train_target[bream_smelt_indexes] # 위의 물고기가 bream인지 smelt인지의 2가지 종류만 들어있음.
## 즉, 우리는 위의 코드를 통해서 bream이나 smelt에 대한 정보만을 추출해 놓게 된 것이다.
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print('앞의 5개의 훈련 데이터에 대한 예측값 : ',lr.predict(train_bream_smelt[:5]))
print('앞의 5개의 훈련 데이터에 대한 확률 : ',lr.predict_proba(train_bream_smelt[:5]))
print('분류 클래스의 순서 : ', lr.classes_)
print('z = a*(Weight) + b*(Length) + c*(Diagonal) + d*(Height) + e*(Width) + f 에서의 a,b,c,d,e : ', lr.coef_)
print('f : ', lr.intercept_)

decisions = lr.decision_function(train_bream_smelt[:5])
print('처음 5개 샘플의 z값 : ',decisions)
from scipy.special import expit
print('양성 클래스에 대한 확률(두 번째 생선일 확률) : ',expit(decisions))

### 로지스틱 회귀를 이용한 다중 분류
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print('\n로지스틱 회기의 다중분류에서\n훈련 데이터 점수 : ',lr.score(train_scaled, train_target))
print('테스트 데이터 점수 : ',lr.score(test_scaled, test_target))
print('\n처음 5개의 예측 값 : ', lr.predict(test_scaled[:5]))
proba = lr.predict_proba(test_scaled[:5])
print('물고기 순서 : ',lr.classes_)
print('\n앞의 5개의 각 샘플들의 물고기 분류 확률 : \n', np.round(proba, decimals=3))
print(lr.coef_.shape, lr.intercept_.shape)

decision = lr.decision_function(test_scaled[:5])
print('각 샘플마다 클래스 별 z값을 구하자.\n', np.round(decision, decimals=2))
from scipy.special import softmax
proba2 = softmax(decision, axis=1)
# axis=1로 하야 각 행(각 샘플)마다의 소프트맥스를 계산하게 된다. 만약 axis매개변수를 지정하지 않으면, 배열 전체에 대한 소프트맥스를 계산한다.(전체 배열의 합이 1이됨 각 행마다 합이 1이 아니라.)
print('\n위의 z값들을 소프트맥스 함수를 이용하여 확률로 변환한 것 : \n', np.round(proba2, decimals=3))
print('\nproba함수를 통해서 나온 예측 확률 값 : \n', np.round(proba, decimals=3), '\n정확히 일치하다.')