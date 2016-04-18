
class MaxPoolingLayer:

    def __init__(self, (width, height, in_depth), pool_width):
        self.width = width
        self.height = height
        self.in_depth = in_depth
        self.pool_width = pool_width
        if width % pool_width != 0 or height % pool_width != 0:
            print("Pool width must be a multiple of width and height")
            exit(1)

    def get_output(self, input):
        if (len(input) != self.width*self.height*self.in_depth):
            print("Invalid input size")
            exit(1)

        out_width = self.width / self.pool_width
        out_height = self.height / self.pool_width

        out = [0] * (out_width * out_height * self.in_depth)

        for x in xrange(0, out_width):
            for y in xrange(0, out_height):
                for i in xrange(0, self.in_depth):
                    max_val = -999999;
                    for p_x in xrange(0, self.pool_width):
                        for p_y in xrange(0, self.pool_width):
                            max_val = max(
                                max,
                                input[i +
                                      (x*self.pool_width+p_x)*self.in_depth +
                                      (y*self.pool_width+p_y)*self.in_depth*self.width
                                ])
                    out[i + x*self.in_depth + y*out_width*self.in_depth] = max_val
        return out

