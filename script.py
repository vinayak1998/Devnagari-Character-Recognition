import tensorflow as tf
import numpy as np
import time
import pandas as pd
import keras
from keras import Sequential
from keras.models import Model
from keras.layers import *
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import sys

tr_data = pd.read_csv(sys.argv[1], header=None)

test_data = pd.read_csv(sys.argv[2], header=None)

Y_train = tr_data[0].values

del tr_data[0]

X_train = tr_data.values

del test_data[0]
X_test = test_data.values

X_train_new = np.zeros((len(X_train), 32, 32, 1))
for i in range(len(X_train)):
    if i % 1000 == 0:
        print(i)
    for a in range(32):
        for b in range(32):
            X_train_new[i][a][b][0] = X_train[i][32 * a + b]

X_test_new = np.zeros((len(X_test), 32, 32, 1))
for i in range(len(X_test)):
    if i % 1000 == 0:
        print(i)
    for a in range(32):
        for b in range(32):
            X_test_new[i][a][b][0] = X_test[i][32 * a + b]

X_train_new /= 255
X_test_new /= 255
num_category = 46
y_train = keras.utils.to_categorical(Y_train, num_category)
input_shape = (32, 32, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_category, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
datagen.fit(X_train_new)
batch_size=86
history = model.fit_generator(datagen.flow(X_train_new,y_train, batch_size=batch_size),
                              epochs = 1, validation_data=(X_train_new, y_train),
                              verbose = 2, steps_per_epoch=X_train_new.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

pred= model.predict(X_test_new)
predictions=np.argmax(pred,axis=1)
np.savetxt(sys.argv[3],predictions)