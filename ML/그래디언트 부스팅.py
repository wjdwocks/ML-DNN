import pandas as pd
wine = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/wine.csv')
wine_input = wine[['pH', 'alcohol', 'sugar']].to_numpy()
wine_target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split, cross_validate
train_input, test_input, train_target, test_target = train_test_split(wine_input, wine_target, test_size=0.2)

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

GradB = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, max_depth=3)
GradBscores = cross_validate(GradB, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(GradBscores['train_score']), np.mean(GradBscores['test_score']))
print('-----------------------------------')

from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier()
scores = cross_validate(hgb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

from sklearn.inspection import permutation_importance
hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10, n_jobs=-1)
print(result.importances_mean)

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
scores = cross_validate(gbr, train_input, train_target, return_train_score=True, n_jobs=-1)
print('-------------------------------')
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

from xgboost import XGBClassifier
xgb = XGBClassifier(learn_rate=0.1, max_depth=6, n_estimators=100, n_jobs=-1)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))