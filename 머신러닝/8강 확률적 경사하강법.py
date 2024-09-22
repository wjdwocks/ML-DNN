import pandas as pd
fish = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fish.csv')
fish_input = fish[['Weight', 'Length', 'Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input, train_target)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
# 여기에서 loss='log'는 손실함수를 지정해줄 수 있는데, log_loss는 로지스틱 손실 함수를 지정해 준것임.
# 또한 max_iter=10에서 전체를 반복할 epoch를 10으로 지정해 준 것임.
sc.fit(train_scaled, train_target)
print('확률적 경사 하강법에 의한 epoch=10의 점수')
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
print()
sc.partial_fit(train_scaled, train_target)
print('확률적 경사 하강법에 의한 epoch=11에서의 점수')
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
print()

import numpy as np
sc = SGDClassifier(loss='log_loss', random_state=42)
train_score=[]
test_score = []
classes = np.unique(train_target)

for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))
    
import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('score')
plt.show()

sc = SGDClassifier(loss='log_loss', tol=None, max_iter=100, random_state=42)
sc.fit(train_scaled, train_target)
print('확률적 경사 하강법에 의한 epoch=100의 점수')
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
print()





