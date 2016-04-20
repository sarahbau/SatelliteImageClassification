import pickle
import layer.Layer as Layer
import layer.Network as Network
import random
import numpy

def norm(val, mean, std):
    return (val - mean)/std

def normalize_vec(vec, stds, means):
    for i, v in enumerate(vec):
        vec[i] = norm(v, means[i], stds[i])
    return vec


data6 = pickle.load(open('../data/image6data_10c.dat', "rb"))['all']
data1 = pickle.load(open('../data/image1data_10c.dat', "rb"))['all']
vals = [[]] * len(data6['residential'][0])

for i in xrange(len(data6['residential'][0])):
    vals[i] = []

for d in data6:
    for ve in data6[d]:
        for i, v in enumerate(ve):
            vals[i].append(v)

stds = []
means = []
for v in vals:
    stds.append(numpy.std(v))
    means.append(numpy.mean(v))

for d in data6:
    for i, ve in enumerate(data6[d]):
        data6[d][i] = normalize_vec(ve, stds, means)

for d in data1:
    for i, ve in enumerate(data1[d]):
        data1[d][i] = normalize_vec(ve, stds, means)
print data1

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