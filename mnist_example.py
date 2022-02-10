import tensorflow
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D
from tensorflow.python.keras import activations

from dl_models import functional_model, MyCustomModel
from my_utils import display_samples

#Using tensorflow.keras.Sequential
seq_model = tensorflow.keras.Sequential(
    [
        Input(shape=(28,28,1)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax'),
    ]
)



if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
    print(f'X_train: {X_train.shape}')
    print(f'y_train: {y_train.shape}')
    print(f'X_test: {X_test.shape}')
    print(f'y_test: {y_test.shape}')

    if False:
        display_samples(X_train, y_train)

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

    model = MyCustomModel()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

    print('\n\nTraining Model....')
    model.fit(X_train, y_train, batch_size=64, epochs=3, validation_split=0.2)
    
    print('\n\nTest Set Predictions....')
    model.evaluate(X_test, y_test, batch_size=64)