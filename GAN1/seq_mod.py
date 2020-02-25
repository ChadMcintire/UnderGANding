import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.datasets import cifar10
from keras.optimizers import Adam

#get data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
NUM_CLASSES = 10

#normalize x values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#turn y into a one-hot vector
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

#make a sequential model, with 3 activation layers
model = Sequential([
    Dense(200, activation = 'relu', input_shape=(32,32,3)),
    Flatten(),
    Dense(150, activation = 'relu'),
    Dense(10, activation = 'softmax'),
])

model.summary()

opt = Adam(lr=.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt,
                metrics=['accuracy'])

model.fit(x_train
          , y_train
          , batch_size = 32
          , epochs =10
          , shuffle = True
          )

model.evaluate(x_test, y_test)
