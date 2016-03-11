from __future__ import print_function
import numpy


class Layer:

    def __init__(self, num_in = None, num_out = None, weight_matrix = None):
        if num_in is not None:
            self.num_in = num_in
        if num_out is not None:
            self.num_out = num_out

        if weight_matrix is None:
            self.weights = numpy.zeros(shape=(num_in + 1, num_out))
        else:
            self.weights = weight_matrix;

    def get_output(self, input):
        return Layer.logistify_array(self.get_outsums(input))

    def get_outsums(self, input):
        in_matrix = numpy.matrix([1] + input)
        return numpy.squeeze(numpy.asarray(in_matrix * self.weights))

    def get_weight(self, in_neuron, out_neuron):
        # Add one to in_neuron because idx zero is the bias for the corresponding out neuron
        return self.weights[in_neuron + 1, out_neuron]

    def get_bias(self, out_neuron):
        return self.weights[0, out_neuron]

    def set_weight(self, in_neuron, out_neuron, weight):
        self.weights[in_neuron + 1, out_neuron] = weight

    def set_bias(self, out_neuron, bias):
        self.weights[0, out_neuron] = bias

    @staticmethod
    def logistic(val):
        return 1.0 / (1.0 + numpy.exp( -val ))

    @staticmethod
    def logistify_array(array):
        for n in range(0, len(array)):
                array[n] = Layer.logistic(array[n])
        return array
