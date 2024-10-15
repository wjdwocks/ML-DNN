from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def createModel(model_fn = []):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu', input_shape=(784,)))
    model.add(keras.layers.Dropout(0.3))
    if model_fn:
        for layer in model_fn:
            model.add(layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

if __name__ == '__main__':
    (train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
    train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2)
    model = createModel([keras.layers.Dense(40, activation='relu')])
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint_cb = keras.callbacks.ModelCheckpoint('2024.10.3.keras', save_best_only=True)
    earlystop_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    model.fit(train_input, train_target, epochs=50, validation_data=(val_input, val_target), callbacks=[checkpoint_cb, earlystop_cb])