import unittest
import layer.Layer as Layer
import layer.Network as Network

class TestNetwork(unittest.TestCase):
    def test_multi_layer_network(self):
        l1 = Layer.Layer(2, 2)
        l2 = Layer.Layer(2, 2)

        l1.set_weight((0, 0), 1)
        l1.set_weight((1, 1,), 2)

        l2.set_weight((0, 1), 1)
        l2.set_weight((1, 0), 2)

        network = Network.Network()
        network.addLayer(l1)
        network.addLayer(l2)

        out = network.get_output([0, 0])
        print (out)

if __name__ == '__main__':
    unittest.main()
