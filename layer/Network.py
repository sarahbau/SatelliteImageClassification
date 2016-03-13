import numpy


class Network:

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        if len(self.layers) > 0:
            if self.layers[len(self.layers)-1].num_out != layer.num_in:
                print "failed to add layer because layer size didn't match"
                exit(1)
        self.layers.append(layer)

    def get_output(self, ins):

        for layer in self.layers:
            ins = layer.get_output(ins)

        return ins

    def back_propagate(self, input, expected_out, learning_rate):

        actual_out = self.get_output(input)

        # calculate sum squared error (diff used as derivative for back-prop)
        diff = numpy.subtract(actual_out, expected_out)
        diffsqr = numpy.multiply(diff, diff)
        error = .5 * numpy.sum(diffsqr)

        cur_der = diff.tolist()

        for layer in reversed(self.layers):
            cur_der = layer.get_reverse_der_sums(cur_der)
            layer.adjust_weights(learning_rate)

        return error
