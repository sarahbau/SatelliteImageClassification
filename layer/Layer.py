import numpy
import random


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

        # last input values into the network, includes the bias value
        self.last_input = [0] * num_in

        # derivative of the last output to the next layer
        self.last_output_der = [0] * num_out

        # the total change in error dE/dwij relative to each output node after back-propagation has occured
        self.backprop_der = [0] * num_out

    def randomize(self, low, high):
        for i in xrange(0, self.num_in + 1):
            for j in xrange(0, self.num_out):
                self.weights[i, j] = random.uniform(low, high)

    def get_output(self, input):
        """
        returns the output of the layer for a given input vector
        :param input: list of elements to input into the layer
        :return: an output from the layer based on weights and evaluation function
        """
        self.last_input = [1] + input
        out_val = Layer.logistify_array(self.get_outsums(input))
        self.last_output_der = Layer.calculate_sig_der_array(out_val)
        return out_val

    def get_reverse_der_sums(self, output):
        """
        used to propagate derivatives backwards through the network, feed in derivatives for the output
        and it will return derivatives at the inputs
        :param output: list (vector) of derivative values from previous layers
        :return:
        """
        self.backprop_der = numpy.multiply(output, self.last_output_der).tolist()
        mid_mat = numpy.matrix(self.backprop_der)
        back_output = (self.weights * mid_mat.transpose()).transpose()
        return numpy.asarray(back_output)[0].tolist()[1:]

    def adjust_weights(self, learning_rate):
        """
        Adjust weights in this layer based on backpropagation derivatives and previous input
        :param learning_rate: rate at which to adjust the weights
        :return:
        """
        for i in xrange(0, self.num_in + 1):
            for j in xrange(0, self.num_out):
                self.weights[i, j] -= learning_rate * self.last_input[i] * self.backprop_der[j]

    def get_outsums(self, input):
        """
        Gets does the bias + matrix multiplication step to calculate sums of inputs after being multiplied by their
        appropriate weights
        :param input: input vector into layer
        :return: sum of input vectors multiplied by weights
        """
        in_matrix = numpy.matrix([1] + input)
        return numpy.asarray(in_matrix * self.weights)[0].tolist()

    def get_weight(self, (in_neuron, out_neuron)):
        # Add one to in_neuron because idx zero is the bias for the corresponding out neuron
        return self.weights[in_neuron + 1, out_neuron]

    def get_bias(self, out_neuron):
        return self.weights[0, out_neuron]

    def set_weight(self, (in_neuron, out_neuron), weight):
        self.weights[in_neuron + 1, out_neuron] = weight

    def set_bias(self, out_neuron, bias):
        self.weights[0, out_neuron] = bias

    @staticmethod
    def logistic(val):
        return 1.0 / (1.0 + numpy.exp( -val ))

    @staticmethod
    def logistify_array(array):
        """
        calculate logistic function on each element of an array
        :param array:
        :return:
        """
        for n in xrange(0, len(array)):
                array[n] = Layer.logistic(array[n])
        return array

    @staticmethod
    def calculate_sig_der_array(array):
        """
        calculate the derivative of each value of an array assuming they are produced from the logistic function
        :param array:
        :return:
        """
        der = [0] * len(array)
        for idx, val in enumerate(array):
            # derivative of logistic function
            der[idx] = val * (1 - val)

        return der