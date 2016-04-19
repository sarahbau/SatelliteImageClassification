import pickle
import layer.Layer as Layer
import layer.Network as Network
import random
import numpy

data6 = pickle.load(open('../data/image6data_10c.dat', "rb"))['all']
data1 = pickle.load(open('../data/image1data_10c.dat', "rb"))['all']

hidden_layer_count = 15
l1 = Layer.Layer(10, hidden_layer_count)
l2 = Layer.Layer(hidden_layer_count, 3)

print ("Initializing weights for layer 1")
l1.randomize(-2, 2)
print ("Initializing weights for layer 2")
l2.randomize(-2, 2)

print ("Starting Network Training")
network = Network.Network()

network.add_layer(l1)
network.add_layer(l2)

iterations = 200000
learning_rate = 1

total_err = 0.0
for i in xrange(iterations):
    exp = None
    class_name = None
    if i%3 == 0:
        exp = [1, 0, 0]
        class_name = 'nature'
    if i%3 == 1:
        exp = [0, 1, 0]
        class_name = 'commercial'
    if i%3 == 2:
        exp = [0, 0, 1]
        class_name = 'residential'

    val = random.choice(data1[class_name]).tolist()
    actual_out = network.get_output(val)
    max = -1
    maxi = 0
    for j in xrange(3):
        if max < actual_out[j]:
            max = actual_out[j]
            maxi = j
    if (maxi == i%3):
        error = 0
    else:
        error = 1
    total_err += error

print (total_err/iterations)

for i in xrange(iterations):
    exp = None
    class_name = None
    if i%3 == 0:
        exp = [1, 0, 0]
        class_name = 'nature'
    if i%3 == 1:
        exp = [0, 1, 0]
        class_name = 'commercial'
    if i%3 == 2:
        exp = [0, 0, 1]
        class_name = 'residential'

    val = random.choice(data6[class_name]).tolist()
    network.back_propagate(val, exp, learning_rate)

total_err = 0.0
for i in xrange(iterations):
    exp = None
    class_name = None
    if i%3 == 0:
        exp = [1, 0, 0]
        class_name = 'nature'
    if i%3 == 1:
        exp = [0, 1, 0]
        class_name = 'commercial'
    if i%3 == 2:
        exp = [0, 0, 1]
        class_name = 'residential'

    val = random.choice(data1[class_name]).tolist()
    actual_out = network.get_output(val)
    max = -1
    maxi = 0
    for j in xrange(3):
        if max < actual_out[j]:
            max = actual_out[j]
            maxi = j
    if (maxi == i%3):
        error = 0
    else:
        error = 1
    total_err += error

print (total_err/iterations)