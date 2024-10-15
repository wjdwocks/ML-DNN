# 데이터 가져오기
import pandas as pd
wine = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/wine.csv')
# 누락된 데이터가 있는지 알아보는 함수
wine.info() # 전부 6497 non-null이므로 누락된 건 없는거로 보인다.
# 누락된 데이터가 있다면 그 행의 데이터를 평균값으로 채우거나, 그 행 자체를 없애버릴 수 있다.

# 각 열(특성)의 평균, 표준편차, 최소, 최대 값, 중간값 등을 알려주는 함수.
print(wine.describe())


wine_input = wine[['alcohol', 'sugar', 'pH']].to_numpy()
wine_target = wine['class'].to_numpy()

# 학습 데이터/테스트 데이터 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(wine_input, wine_target, random_state=42, test_size=0.2)
print('학습 데이터 양 : 테스트 데이터 양 == ', train_input.shape[0], ':' , test_input.shape[0])

# 데이터 정규화 하기
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 로지스틱 회귀를 이용하여 분류하기
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print('학습 데이터 점수 : ', lr.score(train_scaled, train_target))
print('테스트 데이터 점수 : ', lr.score(test_scaled, test_target))

print('z = aw + bx + cy + k 라고 하는 로지스틱 선형 회귀였다면 \n(a, b, c)의 값은 : ', lr.coef_)
print('k의 값은 : ', lr.intercept_)
print('즉, 위에서 나온 z값을 시그모이드 함수에 넣어서 0에 가까우면 0번 클래스, 1에 가까우면 1번 클래스로 선택된다.')
print(lr.predict_proba(test_scaled))

### 결정 트리를 이용하여 학습해보자.

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(train_scaled, train_target)
print('결정 트리를 이용하여 예측한 학습 데이터 점수 vs 테스트 데이터 점수')
print(dt.score(train_scaled, train_target), 'vs',dt.score(test_scaled, test_target))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
# plt.figure(figsize=(10, 7))
# plot_tree(dt)
# plt.show()

### 너무 복잡해서 알아볼 수가 없다.
### 조금 알아보기 쉽게 해보자.
# plt.figure(figsize=(10,7))
# plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
# plt.show()

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print('결정 트리를 이용하여 예측한 학습 데이터 점수 vs 테스트 데이터 점수')
print(dt.score(train_scaled, train_target), 'vs',dt.score(test_scaled, test_target))
# plt.figure(figsize=(20,15))
# plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
# plt.show()

### 특성의 전처리가 필요 없다.
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print('결정 트리를 이용하여 예측한 학습 데이터 점수 vs 테스트 데이터 점수')
print(dt.score(train_input, train_target), 'vs',dt.score(test_input, test_target))
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()


### 결정 트리는 어떤 특성이 가장 중요한 영향을 미쳤는지도 알려준다.
print(dt.feature_importances_)