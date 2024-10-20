import numpy as np
from src.data_preprocessing import load_and_preprocess_data
from src.model import create_model

def train_model():
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()

    model = create_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    model.save('cifar10_model.h5')

if __name__ == "__main__":
    train_model()
