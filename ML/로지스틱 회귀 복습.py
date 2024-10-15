import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/wine.csv', header = 'infer')
train_input = data[['alcohol', 'sugar', 'pH']].to_numpy()
train_target = data['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(train_input, train_target)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))