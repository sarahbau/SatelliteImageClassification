from itertools import chain
import FullyConnectedLayer as Layer


class ConvolutionLayer:

    def __init__(self, (width, height, in_depth), (s_width, s_height), (out_width, out_height, num_features), conv_layer = None):
        if conv_layer is None:
            self.conv_layer = Layer(s_width*s_height*in_depth, num_features)
        else:
            self.conv_layer = conv_layer
            if (conv_layer.num_in != s_width*s_height*in_depth) or (conv_layer.num_out != num_features):
                print "provided convolutional layer was wrong size!"
                exit(1)
        self.width = width
        self.height = height
        self.in_depth = in_depth
        self.s_width = s_width
        self.s_height = s_height
        self.out_width = out_width
        self.out_height = out_height
        self.num_features = num_features

    def randomize(self, low, high):
        self.conv_layer.randomize(low, high)

    def get_convolution_network(self):
        return self.conv_layer

    def get_output(self, input):
        if len(input) != self.width * self.height * self.in_depth:
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
        conv_in_vec = [0] * (self.in_depth * self.s_width * self.s_height)
        for x in xrange(0, self.s_width):
            for y in xrange(0, self.s_height):
                for z in xrange(0, self.in_depth):
                    conv_in_vec[z + x*self.in_depth + y*self.s_width*self.in_depth] = input[z + (x + x_pos)*self.in_depth + (y + y_pos)*self.width*self.in_depth]
        return self.conv_layer.get_output(conv_in_vec)

