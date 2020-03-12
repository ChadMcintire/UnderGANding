from auto_encoder import Autoencoder
import os
from keras.datasets import mnist
import pickle

 
if __name__ == '__main__':
    SECTION = 'vae'
    RUN_ID = '0001'
    DATA_NAME = 'digits'
    RUN_FOLDER = 'run/{}/'.format(SECTION)
    RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

    def load_model(model_class, folder):

        with open(os.path.join(folder, 'params.pkl'), 'rb') as f:
            params = pickle.load(f)

        model = model_class(*params)

        model.load_weights(os.path.join(folder, 'weights/weights.h5'))

        return model
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))

    AE = load_model(Autoencoder, RUN_FOLDER)


