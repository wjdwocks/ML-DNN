import pandas as pd
fish = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fish.csv')
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input, train_target)
train_scale = ss.transform(train_input)
test_scale = ss.transform(test_input)

from sklearn.linear_model import LogisticRegression
import numpy as np
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scale, train_target)
print(lr.score(train_scale, train_target))
print(lr.score(test_scale, test_target))
print(np.round(lr.predict_proba(train_scale[:5]), decimals=3))