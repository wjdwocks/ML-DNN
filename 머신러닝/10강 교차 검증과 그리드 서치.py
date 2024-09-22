import pandas as pd
wine = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/wine.csv')

wine_input = wine[['alcohol', 'sugar', 'pH']].to_numpy()
wine_target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(wine_input, wine_target, test_size=0.2, random_state=42)
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

print(test_input.shape, ' ',sub_input.shape, ' ', val_input.shape)



from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42, max_depth=3)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))
print(dt.score(test_input, test_target))

print('교차 검증')
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)


from sklearn.model_selection import StratifiedKFold
import numpy as np
score = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(score['test_score']))


splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))

# ------------------그리드서치---------------------

from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease':[0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(gs.best_params_)

# 각 매개변수 0.0001 ~ 0.0005에서 나온 평균 교차검증 점수
params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),
          'max_depth' : range(5, 20, 1),
          'min_samples_split' : range(2, 100, 10)}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))
print(gs.score(test_input, test_target))




print('-----------------랜덤 서치--------------------')
from scipy.stats import uniform, randint
rgen = randint(0, 10)
rgen.rvs(10)


params = {'min_impurity_decrease' : uniform(0.0001, 0.001),
          'max_depth' : randint(20,50),
          'min_samples_split' : randint(2,25),
          'min_samples_leaf' : randint(1, 25),}
from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)
print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))
dt = gs.best_estimator_
print(dt.score(test_input, test_target))