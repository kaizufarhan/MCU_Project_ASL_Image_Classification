"""
    models.py
    Created by Adam Kohl
    08.04.2021
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input


def model1(img_h, img_w):
    # Feature extraction
    tf_in = Input(shape=(img_h, img_w, 3), name='image')
    x = Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='relu', padding='VALID')(tf_in)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation='relu', padding='VALID')(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation='relu', padding='VALID')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='VALID')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='VALID')(x)

    # Flatten convolutions for FCN
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)

    # Steering branch
    tf_s = Dense(50, activation='relu')(x)
    tf_s = Dense(10, activation='relu')(tf_s)
    tf_s = Dense(1, activation='linear', name='steering')(tf_s)

    # Throttle branch
    tf_t = Dense(50, activation='relu')(x)
    tf_t = Dense(10, activation='relu')(tf_t)
    tf_t = Dense(1, activation='linear', name='throttle')(tf_t)
    return tf.keras.Model(inputs=tf_in, outputs=[tf_s, tf_t])


def model2(img_h, img_w):
    # Feature extraction
    tf_in = Input(shape=(img_h, img_w, 3), name='image')
    x = Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='relu', padding='VALID')(tf_in)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation='relu', padding='VALID')(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation='relu', padding='VALID')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='VALID')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='VALID')(x)

    # Flatten convolutions for FCN
    x = Flatten()(x)

    # Steering and throttle are on same branch
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    tf_s = Dense(1, activation='linear', name='steering')(x)
    tf_t = Dense(1, activation='linear', name='throttle')(x)
    return tf.keras.Model(inputs=tf_in, outputs=[tf_s, tf_t])
