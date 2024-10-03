from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def model_fn(a_layer = []):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu', input_shape=(784,)))
    if a_layer:
        for layer in a_layer:
            model.add(layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

if __name__ == '__main__':
    (train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
    train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2)
    model = model_fn([keras.layers.Dense(50, activation='relu'), keras.layers.Dense(20, activation='relu')])
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_input, train_target, validation_data=(val_input, val_target), epochs=10)
    print(history.history.keys())