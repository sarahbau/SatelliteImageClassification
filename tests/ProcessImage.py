import layer.Network
from PIL import Image
import pickle


# Processes an image using the specified neural network

image_file = "../images/image.png"
network_file = "../neural_nets/nnet200_40_20000"
output_image = "../images/out.png"

sample_width = 200
w2 = sample_width/2
down_sample = 40

view_width = sample_width / down_sample
num_neurons = view_width * view_width * 3

original_image = Image.open(image_file)
original_pixels = original_image.load()

network = pickle.load(open(network_file, "rb"))

out_image = Image.new('RGB', (original_image.size[0]/100, original_image.size[1]/100), "black")
out_pix = out_image.load()

data = [0] * num_neurons

for x in xrange(sample_width, original_image.size[0] - sample_width, 100):
    print x
    for y in xrange(sample_width, original_image.size[1] - sample_width, 100):
        for x2 in xrange(0, sample_width, down_sample):
            for y2 in xrange(0, sample_width, down_sample):
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
        vals = network.get_output(data)
        out_pix[x/100, y/100] = (int(vals[0]*255), int(vals[1]*255), int(vals[2]*255))

out_image.save(output_image)