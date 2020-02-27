from encoder import Encoder

class Autoencoder():
    def __init__(self,
        input_dim,
        encoder_conv_filters,
        encoder_conv_kernel_size,
        encoder_conv_strides,
        z_dim,
        use_batch_norm = False,
        use_dropout = False,
            ):
        
        self.name = 'autoencoder'
        self.input_dim =  input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.z_dim = z_dim
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self._build()

    def _build(self):
        encode = Encoder(self.input_dim, self.encoder_conv_filters, self.encoder_conv_kernel_size, self.encoder_conv_strides, self.z_dim, self.use_batch_norm, self.use_dropout)
        print(encode.name)
 
if __name__ == '__main__':
    input_dim = (28,28,1)
    encoder_conv_filters = [32,64,64, 64]
    encoder_conv_kernel_size = [3,3,3,3]
    encoder_conv_strides = [1,2,2,1]
    z_dim = 2
    AE = Autoencoder(input_dim,
            encoder_conv_filters,
            encoder_conv_kernel_size,
            encoder_conv_strides,
            z_dim
            )
    print(AE.name)
