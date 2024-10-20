import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Normalize the data
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
    print("Data loaded and preprocessed.")
