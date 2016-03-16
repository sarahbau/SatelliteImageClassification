import Image
import layer.Network as Network
import layer.Layer as Layer
import random
import numpy

print ("Loading original image")
original_image = Image.open("../images/image.png")
original_pixels = original_image.load()

print ("Loading mask image")
mask_image = Image.open("../images/mapping.png")
mask_pixels = mask_image.load()

print original_image.size
print mask_image.size

real_width = 200
down_sample = 40
view_width = real_width / down_sample
num_neurons = view_width * view_width * 3
hidden_layer_count = int(num_neurons * 1)

print ("layer size", hidden_layer_count*num_neurons)

l1 = Layer.Layer(num_neurons, hidden_layer_count)
l2 = Layer.Layer(hidden_layer_count, 3)

print ("Initializing weights for layer 1")
l1.randomize(-2, 2)
print ("Initializing weights for layer 2")
l2.randomize(-2, 2)

print ("Starting Network Training")
network = Network.Network()

network.add_layer(l1)
network.add_layer(l2)
w2 = real_width/2


data = [0] * num_neurons

iterations = 100000

#total error every 100 iterations
total_error = 0
sample_total_error = 0

for i in xrange(0, iterations):
    success = False
    if i % 100 == 99:
        print str(total_error/100) + ", " + str(sample_total_error/100)
        total_error = 0
        sample_total_error = 0

    while not success:
        x = int(random.uniform(real_width, original_image.size[0] - real_width))
        y = int(random.uniform(real_width, original_image.size[1]/2 - real_width))
        mask_c = mask_pixels[x,y][0:3]

        exp = None
        if mask_c == (255, 0, 0):
            exp = [1, 0, 0]     # houses
            if i % 3 != 0:
                continue
        elif mask_c == (0, 255, 0):
            exp = [0, 1, 0]     # trees
            if i % 3 != 1:
                continue
        elif mask_c == (0, 0, 255):
            exp = [0, 0, 1]     # industry
            if i % 3 != 2:
                continue

        if exp is not None:
            mask_r = mask_pixels[x+w2, y][0:3]
            mask_l = mask_pixels[x-w2, y][0:3]
            mask_u = mask_pixels[x, y-w2][0:3]
            mask_d = mask_pixels[x, y+w2][0:3]

            if all(mask_c == x for x in (mask_r, mask_l, mask_u, mask_d)):
                for x2 in xrange(0, real_width, down_sample):
                    for y2 in xrange(0, real_width, down_sample):
                        avg_col = [0, 0, 0]
                        for x3 in xrange(0, down_sample):
                            for y3 in xrange(0, down_sample):
                                col = original_pixels[x+x2-w2+x3, y+y2-w2+y3]
                                avg_col[0] += col[0]
                                avg_col[1] += col[1]
                                avg_col[2] += col[2]
                        avg_col[0] /= (down_sample*down_sample)
                        avg_col[1] /= (down_sample*down_sample)
                        avg_col[2] /= (down_sample*down_sample)

                        idx = 3*(x2/down_sample + y2/down_sample*view_width)
                        data[idx] = avg_col[0]/255.0
                        data[idx+1] = avg_col[1]/255.0
                        data[idx+2] = avg_col[2]/255.0
                err = network.back_propagate(data, exp, .1)
                #print (exp, err)
                total_error += err
                success = True

    success = False
    while not success:
        x = int(random.uniform(real_width, original_image.size[0] - real_width))
        y = int(random.uniform(original_image.size[1]/2 + real_width, original_image.size[1] - real_width))
        mask_c = mask_pixels[x,y][0:3]

        exp = None
        if mask_c == (255, 0, 0):
            exp = [1, 0, 0]     # houses
            if i % 3 != 0:
                continue
        elif mask_c == (0, 255, 0):
            exp = [0, 1, 0]     # trees
            if i % 3 != 1:
                continue
        elif mask_c == (0, 0, 255):
            exp = [0, 0, 1]     # industry
            if i % 3 != 2:
                continue

        if exp is not None:
            mask_r = mask_pixels[x+w2, y][0:3]
            mask_l = mask_pixels[x-w2, y][0:3]
            mask_u = mask_pixels[x, y-w2][0:3]
            mask_d = mask_pixels[x, y+w2][0:3]

            if all(mask_c == x for x in (mask_r, mask_l, mask_u, mask_d)):
                for x2 in xrange(0, real_width, down_sample):
                    for y2 in xrange(0, real_width, down_sample):
                        avg_col = [0, 0, 0]
                        for x3 in xrange(0, down_sample):
                            for y3 in xrange(0, down_sample):
                                col = original_pixels[x+x2-w2+x3, y+y2-w2+y3]
                                avg_col[0] += col[0]
                                avg_col[1] += col[1]
                                avg_col[2] += col[2]
                        avg_col[0] /= (down_sample*down_sample)
                        avg_col[1] /= (down_sample*down_sample)
                        avg_col[2] /= (down_sample*down_sample)

                        idx = 3*(x2/down_sample + y2/down_sample*view_width)
                        data[idx] = avg_col[0]/255.0
                        data[idx+1] = avg_col[1]/255.0
                        data[idx+2] = avg_col[2]/255.0
                actual_out = network.get_output(data)
                # calculate sum squared error (diff used as derivative for back-prop)
                diff = numpy.subtract(actual_out, exp)
                diffsqr = numpy.multiply(diff, diff)
                error = .5 * numpy.sum(diffsqr)

                sample_total_error += error
                success = True
