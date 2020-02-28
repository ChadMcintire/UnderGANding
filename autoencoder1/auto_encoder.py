from encoder import Encoder
from decoder import Decoder
from keras.models import Model


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
        use_batch_norm = False,
        use_dropout = False,
            ):
        
        self.name = 'autoencoder'
        self.input_dim =  input_dim
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
                         self.use_dropout
                        )

        decode = Decoder(
                         self.decoder_conv_filters,
                         self.decoder_conv_kernel_size,
                         self.decoder_conv_strides,
                         self.z_dim,
                         encode.shape_before_flattening,                
                )

        model_input = encode.encoder_input
        model_output = decode.decoder(encode.encoder_output)

        self.model = Model(model_input, model_output)

 
if __name__ == '__main__':
    input_dim = (28,28,1)
    encoder_conv_filters = [32,64,64, 64]
    encoder_conv_kernel_size = [3,3,3,3]
    encoder_conv_strides = [1,2,2,1]
    decoder_conv_t_filters = [64,64,32,1]
    decoder_conv_t_kernel_size = [3,3,3,3]
    decoder_conv_t_strides = [1,2,2,1]
    z_dim = 2

    AE = Autoencoder(input_dim,
            encoder_conv_filters,
            encoder_conv_kernel_size,
            encoder_conv_strides,
            decoder_conv_t_filters,
            decoder_conv_t_kernel_size,
            decoder_conv_t_strides, 
            z_dim,
            )
