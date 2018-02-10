# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam ,RMSprop


labels = pd.read_csv('input/labels.csv')
imgs = []
Y = labels['breed']
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)

Y_categorical = to_categorical(Y_encoded)

for key, value in labels.iterrows():
    image = load_img('input/train/{0}.jpg'.format(value['id']), target_size=(60, 60))
    image = img_to_array(image)
    imgs.append(image)

X = np.array(imgs, dtype='float32')


# =============================================================================
# X_train, X_test, y_train, y_test = train_test_split(X, Y_categorical, test_size=0.2, random_state=123123)
# =============================================================================


def larger_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(60, 60, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(120, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = larger_model()
# Fit the model
model.fit(X, Y_categorical, epochs=100, batch_size=200, verbose=1, validation_split=0.2)



