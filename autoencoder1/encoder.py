from keras.layers import Input, Conv2D, LeakyReLU, Flatten, Dense
from keras.models import Model
from keras import backend as K


#function api for an encoder
class Encoder():
    def __init__(self,
                 input_dim,
                 encoder_conv_filters,
                 encoder_conv_kernel_size,
                 encoder_conv_strides,
                 z_dim, 
                 use_batch_norm=False,
                 use_dropout=False,
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
        self.layer_setup()

    def layer_setup(self):
        encoder_input = Input(shape=self.input_dim, name='encoder_input')
        self.encoder_input = encoder_input
        x = encoder_input

        for i in range(self.num_layers):
            conv_layer = Conv2D(
                filters = self.encoder_conv_filters[i]
                , kernel_size = self.encoder_conv_kernel_size[i]
                , strides = self.encoder_conv_strides[i]
                , padding = 'same'
                , name = 'encoder_conv_' + str(i)
                )

            x = conv_layer(x)
    
            x = LeakyReLU()(x)
    
            if self.use_batch_norm:
                x = BatchNormalization()(x)
    
            if self.use_dropout:
                x = Dropout(rate = 0.25)(x)

        self.shape_before_flattening = K.int_shape(x)[1:]
        x = Flatten()(x)
        self.encoder_output = Dense(self.z_dim, name='encoder_output')(x)
        self.encoder = Model(encoder_input, self.encoder_output)

        print("Encoder Summary")
        self.encoder.summary()
