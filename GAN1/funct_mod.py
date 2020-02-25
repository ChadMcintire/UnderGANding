import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.models import Model
import keras 
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
input_layer = Input(shape=(32, 32, 3))

#flatten takes the input vector and makes it a 
#1-d vector 32*32*3 = 3072
x = Flatten()(input_layer)

x = Dense(units=200, activation = 'relu')(x)
x = Dense(units=150, activation = 'relu')(x)
output_layer = Dense(units=10, activation = 'softmax')(x)

model = Model(input_layer, output_layer)

print("model summary")

model.summary()

print("model plot")

keras.utils.plot_model(model, 'my_first_model_with_shape_info.png', show_shapes=True)

opt = Adam(lr=.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt,
                metrics=['accuracy'])

model.fit(x_train
          , y_train
          , batch_size = 200
          , epochs =10
          , shuffle = True
          )

model.evaluate(x_test, y_test)

#save the model
model.save('model1.h5')
del model

new_model = keras.models.load_model('path_to_my_model.h5')
