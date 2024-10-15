import pandas as pd
data = pd.read_csv("https://raw.githubusercontent.com/rickiepark/hg-mldl/master/wine.csv")

input_data = data[['alcohol', 'sugar', 'pH']].to_numpy()
target_data = data['class'].to_numpy()

print(input_data[:5])
print(target_data[:5])

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(input_data, target_data)



from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth= 3)
dt.fit(train_input, train_target)
print(dt.score(test_input, test_target))

# 교차 검증에 대한것 다시 공부.

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input, train_target)
scaled_train = ss.transform(train_input)
scaled_test = ss.transform(test_input)

from sklearn.linear_model import LogisticRegressionCV
ls = LogisticRegressionCV()
ls.fit(scaled_train, train_target)
print(ls.score(scaled_test, test_target))


