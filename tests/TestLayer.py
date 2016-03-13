import unittest
import layer.Layer as Layer


class TestLayer(unittest.TestCase):
    def test_simple_network(self):
        network = Layer.Layer(2, 2);
        network.set_weight((0, 0), 1)
        network.set_weight((0, 1), 2)
        network.set_weight((1, 1), 3)
        network.set_weight((1, 0), 4)

        #test simple network with no biases
        sums = network.get_outsums([0, 1])
        self.assertTrue((sums==[4, 3]).all())
        sums = network.get_outsums([1, 0])
        self.assertTrue((sums==[1, 2]).all())
        sums = network.get_outsums([1, 1])
        self.assertTrue((sums==[5, 5]).all())

        #add some biases
        network.set_bias(0, 1)
        network.set_bias(1, -1)
        sums = network.get_outsums([0, 1])
        self.assertTrue((sums==[5, 2]).all())
        sums = network.get_outsums([1, 0])
        self.assertTrue((sums==[2, 1]).all())
        sums = network.get_outsums([1, 1])
        self.assertTrue((sums==[6, 4]).all())
        print(sums)
        sums = network.get_output([1, 1])
        print(sums)
        self.assertTrue((sums>[.99, .98]).all())


if __name__ == '__main__':
    unittest.main()
