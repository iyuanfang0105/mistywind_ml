from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def my_cnn():
    # Define model
    model = keras.Sequential()
    model.add(layers.Convolution2D(16, (3, 3), padding='same', input_shape=(128, 128, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Convolution2D(32, (3, 3), padding='same', activation= 'relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Convolution2D(64, (3, 3), padding='same', activation= 'relu'))
    model.add(layers.MaxPooling2D(pool_size =(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(100, activation='softmax'))
    # Train model
    # adam = tf.train.AdamOptimizer()
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=adam,
    #               metrics=['top_k_categorical_accuracy'])
    print(model.summary())
    return model


if __name__ == '__main__':
    my_cnn()
    