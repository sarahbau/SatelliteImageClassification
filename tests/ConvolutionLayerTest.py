import unittest
import layer.ConvolutionLayer as ConvolutionLayer
import layer.Layer as Layer

class MyTestCase(unittest.TestCase):
    def test_simple_convolution(self):
        l = Layer.Layer(8, 3)
        l.randomize(-1, 1)
        testLayer = ConvolutionLayer.ConvolutionLayer((4, 4, 2), (2, 2), (2, 2, 3), l)
        vec = [1, 1,   1, 1,   1, 1,   1, 1,

               2, 2,   3, 3,   1, 1,   0, 0,

               0, 0,   0, 0,   0, 1,   1, 0,

               1, 1,   2, 2,   0, 1,   1, 0]

        print(testLayer.get_output(vec));


if __name__ == '__main__':
    unittest.main()
