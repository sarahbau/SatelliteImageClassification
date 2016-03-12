import numpy


class Network:

    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        if len(self.layers) > 0:
            if self.layers[len(self.layers)-1].num_out != layer.num_in:
                print "failed to add layer because layer size didn't match"
                exit(1)
        self.layers.append(layer)

    def get_output(self, input):
        for layer in self.layers:
            input = layer.get_output(input)

        return input

