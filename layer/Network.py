import numpy


class Network:

    def __init__(self):
        self.layers = []
        self.lastInputs = []
        self.lastInputsDer = []
        self.lastOutput = []
        self.lastOutputDer = []

    def add_layer(self, layer):
        if len(self.layers) > 0:
            if self.layers[len(self.layers)-1].num_out != layer.num_in:
                print "failed to add layer because layer size didn't match"
                exit(1)
        self.layers.append(layer)

    def get_output(self, ins):
        self.lastInputs = []
        self.lastInputsDer = []
        self.lastOutput = []
        self.lastOutputDer = []

        for layer in self.layers:
            self.lastInputs.append(ins)
            ins = layer.get_output(ins)

        self.lastOutput = ins

        self.calculate_derivatives()
        return ins

    # def train_backpropagation(self, learning_rate):



    def calculate_derivatives(self):
        self.lastOutputDer = Network.calculate_sig_der_array(self.lastOutput)
        self.lastInputsDer = []
        for idx, ins in enumerate(self.lastInputs):
            self.lastInputsDer.append(Network.calculate_sig_der_array(ins))

    @staticmethod
    def calculate_sig_der_array(array):
        der = [0] * len(array)
        for idx, val in enumerate(array):
            der[idx] = val * (1 - val)

        return der



