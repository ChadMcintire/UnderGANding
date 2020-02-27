from encoder import Encoder
import unittest

class TestEncoder(unittest.TestCase):
    def setUp(self):
        input_dim = (28,28,1)
        encoder_conv_filters = [32,64,64, 64]
        encoder_conv_kernel_size = [3,3,3,3]
        encoder_conv_strides = [1,2,2,1]
        z_dim = 2
        self.EC = Encoder(input_dim,
                encoder_conv_filters,
                encoder_conv_kernel_size,
                encoder_conv_strides,
                z_dim
                )

    def test_input_tuple(self):
        tup = ()
        self.assertEqual(type(self.EC.input_dim), type(tup))

    def test_input_size_3(self):
        size = 3
        length_input = len(self.EC.input_dim)
        self.assertEqual(length_input, size)

    def test_filters_kernelsize_strides_lists(self):
        l1 = isinstance(self.EC.encoder_conv_filters, list)
        l2 = isinstance(self.EC.encoder_conv_kernel_size, list)
        l3 = isinstance(self.EC.encoder_conv_strides, list)
        truth_values = [l1,l2,l3]
        self.assertTrue(all(truth_values))

    def test_filter_kernelsize_strides_same_size(self):
        l1 = len(self.EC.encoder_conv_filters)
        l2 = len(self.EC.encoder_conv_kernel_size)
        l3 = len(self.EC.encoder_conv_strides)
        check_for_repeat = set([l1, l2, l3])
        self.assertEqual(len(check_for_repeat), 1)

    def test_preflatten_shape(self):
        print("stuff\n\n", self.EC.shape_before_flattening)

if __name__ == '__main__':
    unittest.main()

