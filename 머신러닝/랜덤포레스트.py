import pandas as pd
import numpy as np
wine = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/wine.csv')

wine_input = wine[['alcohol', 'sugar', 'pH']].to_numpy()
print(wine_input[:5])
wine_target = wine['class'].to_numpy()
print(wine_target[:5])

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(wine_input, wine_target, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

rf = RandomForestClassifier(max_depth=5)
from sklearn.model_selection import StratifiedKFold
scores = cross_validate(rf, train_input, train_target, cv=StratifiedKFold(n_splits=10, shuffle=True), return_train_score=True)
print('학습 점수 vs 검증 점수 : ',np.round(np.mean(scores['train_score']), 3), np.round(np.mean(scores['test_score']), 3))

### oob_score
rf = RandomForestClassifier(oob_score=True)
scores = cross_validate