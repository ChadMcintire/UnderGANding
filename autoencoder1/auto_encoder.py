from encoder import Encoder
from decoder import Decoder
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.datasets import mnist
import os
from callbacks import CustomCallback, step_decay_schedule
from keras.callbacks import ModelCheckpoint



class Autoencoder():
    def __init__(self,
                 input_dim,
                 encoder_conv_filters,
                 encoder_conv_kernel_size,
                 encoder_conv_strides,
                 decoder_conv_filters,
                 decoder_conv_kernel_size,
                 decoder_conv_strides,
                 z_dim,
                 use_batch_norm=False,
                 use_dropout=False,
                 ):

        self.name = 'autoencoder'
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_filters = decoder_conv_filters
        self.decoder_conv_kernel_size = decoder_conv_kernel_size
        self.decoder_conv_strides = decoder_conv_strides
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self._build()

    def _build(self):
        encode = Encoder(self.input_dim,
                         self.encoder_conv_filters,
                         self.encoder_conv_kernel_size,
                         self.encoder_conv_strides,
                         self.z_dim,
                         self.use_batch_norm,
                         self.use_dropout,
                         )

        decoder = Decoder(
            self.decoder_conv_filters,
            self.decoder_conv_kernel_size,
            self.decoder_conv_strides,
            self.z_dim,
            encode.shape_before_flattening,
        )
        self.decoder = decoder.__dict__['decoder']
        model_input = encode.encoder_input
        model_output = decoder.decoder(encode.encoder_output)

        self.model = Model(model_input, model_output)

        #print("\nAutoencoder Summary\n")
        self.model.summary()

        #print("object keys")

        #print(self.__dict__.keys())

    def compile(self, learning_rate):
        self.learning_rate = learning_rate
        optimizer = Adam(lr=learning_rate)

        def r_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

        self.model.compile(optimizer=optimizer, loss=r_loss)

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=100, initial_epoch=0, lr_decay=1):

        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only = True, verbose=1)

        callbacks_list = [checkpoint2, custom_callback, lr_sched]


        self.model.fit(
            x_train
            , x_train
            , batch_size = batch_size
            , shuffle = True
            , epochs = epochs
            , initial_epoch = initial_epoch
            , callbacks = callbacks_list
        )

    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.encoder, to_file=os.path.join(run_folder ,'viz/encoder.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.decoder, to_file=os.path.join(run_folder ,'viz/decoder.png'), show_shapes = True, show_layer_names = True)
        

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save(self, folder):
        
        if not os.path.exists(folder):
                    os.makedirs(folder)
                    os.makedirs(os.path.join(folder, 'viz'))
                    os.makedirs(os.path.join(folder, 'weights'))
                    os.makedirs(os.path.join(folder, 'images'))
        
        #dump takes the object then the file
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim,
                self.encoder_covn_filters,
                self.encoder_conv_kernel_size,
                self.encoder_conv_strides,
                self.decoder_conv_filters,
                self.decoder_conv_kernel_size,
                self.decoder_conv_strides,
                self.z_dim,
                self.use_batch_norm,
                self.use_dropout,
                ], f)

        self.plot_model_(folder)



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
