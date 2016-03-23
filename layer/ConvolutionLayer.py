import numpy
import random
from itertools import chain
import FullyConnectedLayer as Layer


class ConvolutionLayer:

    def __init__(self, width, height, s_width, s_height, out_width, out_height, num_features):
        self.conv_layer = Layer(s_width*s_height, num_features)
        self.width = width
        self.height = height
        self.s_width = s_width
        self.s_height = s_height
        self.out_width = out_width
        self.out_height = out_height
        self.num_features = num_features

    def randomize(self, low, high):
        self.conv_layer.randomize(low, high)

    def get_output(self, input):
        if len(input) != self.width * self.height:
            print "invalid input size"
            exit(1)

        output = [None] * (self.out_width * self.out_height)
        x_step = float(self.width) / self.s_width
        y_step = float(self.height) / self.s_height

        for x in xrange(0, self.out_width):
            for y in xrange(0, self.out_height):
                output[x + y*self.out_width] = self.get_feature_vector(input, (int(x_step * x), int(y_step * y)))

        return list(chain.from_iterable(output))

    def get_feature_vector(self, input, (x_pos, y_pos)):
        conv_in_vec = [0] * (self.s_width * self.s_height)
        for x in xrange(0, self.s_width):
            for y in xrange(0, self.s_height):
                conv_in_vec[x + y*self.s_width] = input[x + x_pos + (y + y_pos)*self.width]

        return self.conv_layer.get_output(conv_in_vec)

