from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2)
model = keras.models.load_model('2024.10.3.keras')
model.evaluate(val_input, val_target)