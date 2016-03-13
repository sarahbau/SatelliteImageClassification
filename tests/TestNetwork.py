import unittest
import layer.Layer as Layer
import layer.Network as Network
import random


class TestNetwork(unittest.TestCase):
    def test_multi_layer_network(self):
        l1 = Layer.Layer(2, 2)
        l2 = Layer.Layer(2, 2)

        l1.set_weight((0, 0), 1)
        l1.set_weight((1, 1,), 2)

        l2.set_weight((0, 1), 1)
        l2.set_weight((1, 0), 2)

        network = Network.Network()
        network.add_layer(l1)
        network.add_layer(l2)

        out = network.get_output([0, 0])
        # print (out)

    def test_backpropagation_single_layer(self):
        l1 = Layer.Layer(2, 1)

        l1.set_weight((0, 0), .1)
        l1.set_weight((1, 0), -.1)
        l1.set_bias(0, .1)

        #nand
        dataset = [
            ([0, 0], [1]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
                   ]
        network = Network.Network()
        network.add_layer(l1)

        for i in range(0, 10000):
            row = random.choice(dataset)
            network.back_propagate(row[0], row[1], 1)

        self.assertTrue(network.get_output([0, 0])[0] > .9)
        self.assertTrue(network.get_output([1, 0])[0] > .9)
        self.assertTrue(network.get_output([0, 1])[0] > .9)
        self.assertTrue(network.get_output([1, 1])[0] < .1)

    def test_backpropagation_two_layer(self):
        l1 = Layer.Layer(2, 5)
        l2 = Layer.Layer(5, 1)
        l1.randomize(-.1, .1)
        l2.randomize(-.1, .1)

        #xor
        dataset = [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
                   ]
        network = Network.Network()
        network.add_layer(l1)
        network.add_layer(l2)

        for i in range(0, 10000):
            row = random.choice(dataset)
            network.back_propagate(row[0], row[1], 1)

        self.assertTrue(network.get_output([0, 0])[0] < .1)
        self.assertTrue(network.get_output([1, 0])[0] > .9)
        self.assertTrue(network.get_output([0, 1])[0] > .9)
        self.assertTrue(network.get_output([1, 1])[0] < .1)

if __name__ == '__main__':
    unittest.main()
