from tensorflow import keras
import numpy as np

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)
#########################################################
#import matplotlib.pyplot as plt                        #
#fig, axs = plt.subplots(1, 10, figsize=(10, 10))       #
#for i in range(10):                                    #
#    axs[i].imshow(train_input[i], cmap='gray')         #
#    axs[i].axis('off')                                 #
#plt.show()                                             #
#########################################################

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_input1d = train_input.reshape(-1, 28*28)
test_input1d = test_input.reshape(-1, 28*28)

ss.fit(train_input1d)
train_scaled = ss.transform(train_input1d)
test_scaled = ss.transform(test_input1d)

# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(max_iter = 1000)
# lr.fit(train_scaled, train_target)
# print(lr.score(test_scaled, test_target))
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
sc = SGDClassifier(loss='log_loss', max_iter=5)
scores = cross_validate(sc, train_scaled, train_target, cv=5, n_jobs=-1)
print(np.mean(scores['test_score']))


## 딥러닝 모델로 학습해보기
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2)


model = keras.Sequential()
model.add(keras.layers.Dense(10, activation='softmax', input_shape=(784,)))
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)
model.summary()