import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/wine.csv')

input_data = data[['alcohol', 'sugar', 'pH']].to_numpy()
target_data = data['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(input_data, target_data, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
max_depths = [3, 5, 7, 9]
for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth = depth)
    dt.fit(train_input, train_target)
    print(dt.score(train_input, train_target), 'vs', dt.score(test_input, test_target))