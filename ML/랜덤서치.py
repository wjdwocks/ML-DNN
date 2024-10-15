import pandas as pd
wine = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/wine.csv')

wine_input = wine[['alcohol', 'sugar', 'pH']].to_numpy()
wine_target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(wine_input, wine_target, test_size=0.2)


from scipy.stats import uniform, randint
rgen = randint(0, 10)
rgen.rvs(10)

params = {'min_impurity_decrease' : uniform(0.0001, 0.001),
          'max_depth' : randint(20, 50),
          'min_samples_split' : randint(2, 25),
          'min_samples_leaf' : randint(1, 25)}

from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
rs = RandomizedSearchCV(DecisionTreeClassifier(), params, n_iter=100, n_jobs=-1, cv=5)

rs.fit(train_input, train_target)

dt = rs.best_estimator_
print('랜덤 서치로 결정된 최적의 결정트리', dt.score(test_input, test_target))


from sklearn.model_selection import GridSearchCV
import numpy as np
gs = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid={
    'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),
    'max_depth' : range(5, 20, 1),
    'min_samples_split' : range(2, 100, 10)
}, n_jobs=-1)

gs.fit(train_input, train_target)
dt = gs.best_estimator_
print('그리드 서치로 결정된 점수 : ',dt.score(test_input, test_target))