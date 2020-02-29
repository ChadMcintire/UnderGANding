from encoder import Encoder
from decoder import Decoder
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.datasets import mnist


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

        print("\nAutoencoder Summary\n")
        self.model.summary()

    def compile(self, learning_rate):
        optimizer = Adam(lr=learning_rate)
        
        def r_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

        self.model.compile(optimizer=optimizer, loss = r_loss)

    def train(self,x_train, batch_size, epochs, initial_epoch ):
            self.model.fit(     
            x_train,
            x_train,
            batch_size = batch_size,
            shuffle = True,
            epochs = epochs,
            initial_epoch = initial_epoch,
        )

 
if __name__ == '__main__':
    input_dim = (28,28,1)
    encoder_conv_filters = [32,64,64, 64]
    encoder_conv_kernel_size = [3,3,3,3]
    encoder_conv_strides = [1,2,2,1]
    decoder_conv_t_filters = [64,64,32,1]
    decoder_conv_t_kernel_size = [3,3,3,3]
    decoder_conv_t_strides = [1,2,2,1]
    z_dim = 2
    lr = 0.0005

    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32
    INITIAL_EPOCH = 0



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

    AE.compile(lr)


    
    AE.train(     
    x_train[:1000],
    batch_size = BATCH_SIZE,
    epochs = 200,
    initial_epoch = INITIAL_EPOCH,
    ) 

    print('\n# Evaluate on test data')
    results = AE.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)
