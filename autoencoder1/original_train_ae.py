import os
from keras.datasets import mnist
from auto_encoder import Autoencoder

if __name__ == '__main__':
    input_dim = (28, 28, 1)
    encoder_conv_filters = [32, 64, 64, 64]
    encoder_conv_kernel_size = [3, 3, 3, 3]
    encoder_conv_strides = [1, 2, 2, 1]
    decoder_conv_t_filters = [64, 64, 32, 1]
    decoder_conv_t_kernel_size = [3, 3, 3, 3]
    decoder_conv_t_strides = [1, 2, 2, 1]
    z_dim = 2
    lr = 0.0005

    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32
    INITIAL_EPOCH = 0

    SECTION = 'vae'
    RUN_ID = '0001'
    DATA_NAME = 'digits'
    RUN_FOLDER = 'run/{}/'.format(SECTION)
    RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

    if not os.path.exists("run"):
        os.mkdir('run')

    if not os.path.exists(os.path.join('run','vae')):
        os.mkdir(os.path.join('run','vae'))


    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
        os.mkdir(os.path.join(RUN_FOLDER, 'images'))
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))

    AE = Autoencoder(input_dim,
                     encoder_conv_filters,
                     encoder_conv_kernel_size,
                     encoder_conv_strides,
                     decoder_conv_t_filters,
                     decoder_conv_t_kernel_size,
                     decoder_conv_t_strides,
                     z_dim,
                     )

    #print(AE.__dict__.keys())
    #print(AE.__dict__['decoder'])
    #print(AE.__dict__.values())
    AE.compile(lr)

    AE.train(
        x_train[:1000]
        , batch_size = BATCH_SIZE
        , epochs = 200
        , run_folder = RUN_FOLDER
        , initial_epoch = INITIAL_EPOCH
    )

    AE.save(RUN_FOLDER)
