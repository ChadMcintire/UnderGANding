from keras.layers import Input, Conv2D, LeakyReLU, Flatten, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import utils


# function api for an encoder
class Encoder():
    def __init__(self,
                 input_dim,
                 encoder_conv_filters,
                 encoder_conv_kernel_size,
                 encoder_conv_strides,
                 z_dim,
                 use_batch_norm=False,
                 use_dropout=False,
                 variation=True,
                 ):
        self.input_dim = input_dim
        self.name = 'encoder'
        self.num_layers = len(encoder_conv_filters)
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.z_dim = z_dim
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.variation = variation
        self.layer_setup()

    def layer_setup(self):
        encoder_input = Input(shape=self.input_dim, name='encoder_input')
        self.encoder_input = encoder_input
        x = encoder_input

        for i in range(self.num_layers):
            conv_layer = Conv2D(
                filters=self.encoder_conv_filters[i],
                kernel_size=self.encoder_conv_kernel_size[i],
                strides=self.encoder_conv_strides[i],
                padding='same',
                name='encoder_conv_' + str(i),
            )

            x = conv_layer(x)

            x = LeakyReLU()(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        self.shape_before_flattening = K.int_shape(x)[1:]
        x = Flatten()(x)

        if self.variation:
            self.mu = Dense(self.z_dim, name='mu')(x)
            self.log_var = Dense(self.z_dim, name='log_var')(x)

            encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

            def sampling(args):
                mu, log_var = args
                epsilon = K.random_normal(
                    shape=K.shape(mu), mean=0., stddev=1.)
                sigma = K.exp(log_var/2)
                return mu + sigma * epsilon
            self.encoder_output = Lambda(sampling, name='encoder_output')([
                self.mu, self.log_var])

        else:
            self.encoder_output = Dense(self.z_dim, name='encoder_output')(x)

        self.encoder = Model(encoder_input, self.encoder_output)
        print("Encoder Summary")
        self.encoder.summary()

        utils.plot_model(self.encoder, 'encoder_model.png',
                         show_layer_names=True, show_shapes=True, rankdir='TB')
