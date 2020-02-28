from keras.layers import Input, Conv2D, LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from keras.models import Model
from keras import backend as K
import numpy as np

class Decoder():
    def __init__(self,
                 decoder_conv_filters,
                 decoder_conv_kernel_size,
                 decoder_conv_strides,
                 z_dim,
                 flattened_shape,
                 use_batch_norm=False,
                 use_dropout=False,
                 ):
        self.name = 'decoder'
        self.num_layers = len(decoder_conv_filters)
        self.decoder_conv_filters = decoder_conv_filters
        self.decoder_conv_kernel_size = decoder_conv_kernel_size
        self.decoder_conv_strides = decoder_conv_strides
        self.z_dim = z_dim
        self.shape_before_flattening = flattened_shape
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.layer_setup()

    def layer_setup(self):
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
        x = Dense(np.prod(self.shape_before_flattening))(decoder_input)
        x = Reshape(self.shape_before_flattening)(x)

        for i in range(self.num_layers):
            conv_t_layer = Conv2DTranspose(
                filters = self.decoder_conv_filters[i]
                , kernel_size = self.decoder_conv_kernel_size[i]
                , strides = self.decoder_conv_strides[i]
                , padding = 'same'
                , name = 'decoder_conv_' + str(i)
                )

            x = conv_t_layer(x)

            if i < self.num_layers - 1:
                x = LeakyReLU()(x)

                if self.use_batch_norm:
                    x = BatchNormalization()(x)

                if self.use_dropout:
                    x = Dropout(rate = 0.25)(x)
            else:
                x = Activation('sigmoid')(x)

        decoder_output = x
        self.decoder = Model(decoder_input, decoder_output)
        
