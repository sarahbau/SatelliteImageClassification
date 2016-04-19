import sift.vlfeat as sift
from sklearn.decomposition import PCA
import os
from random import shuffle
import layer.Network as Network
import layer.Layer as Layer
import numpy

def exctractImgNames():

    # com = 0
    # nat = 1
    # res = 2
    # oth = 3

    imgs_file_names = {}
    for i in os.listdir("../images/Classified/1/Commercial/"):
        imgs_file_names["../images/Classified/1/Commercial/"+i] = 0

    for i in os.listdir("../images/Classified/1/Nature/"):
        imgs_file_names["../images/Classified/1/Nature/"+i] = 1

    for i in os.listdir("../images/Classified/1/Residential/"):
        imgs_file_names["../images/Classified/1/Residential/"+i] = 2

    for i in os.listdir("../images/Classified/2/Commercial/"):
        imgs_file_names["../images/Classified/2/Commercial/"+i] = 0

    for i in os.listdir("../images/Classified/2/Nature/"):
        imgs_file_names["../images/Classified/2/Nature/"+i] = 1

    for i in os.listdir("../images/Classified/2/Residential/"):
        imgs_file_names["../images/Classified/2/Residential/"+i] = 2

    for i in os.listdir("../images/Classified/6/Commercial/"):
        imgs_file_names["../images/Classified/6/Commercial/"+i] = 0

    for i in os.listdir("../images/Classified/6/Nature/"):
        imgs_file_names["../images/Classified/6/Nature/"+i] = 1

    for i in os.listdir("../images/Classified/6/Residential/"):
        imgs_file_names["../images/Classified/6/Residential/"+i] = 2

    for i in os.listdir("../images/Classified/6/Other/"):
        imgs_file_names["../images/Classified/6/Other/"+i] = 3

    return imgs_file_names

def createNetwork(in_neurons, hidden_layer_count, n_outs):
    network = Network.Network()
    l1 = Layer.Layer(in_neurons, hidden_layer_count)
    l2 = Layer.Layer(hidden_layer_count, n_outs)
    l1.randomize(-1, 1)
    l2.randomize(-1, 1)
    network.add_layer(l1)
    network.add_layer(l2)
    return network

def trainNet(file_names,dict,network,rate,n_comp):
    errors = []
    for i in file_names:
        sift.process_image(i, 'tmp.sift')
        l, d = sift.read_features_from_file('tmp.sift')
        pca = PCA(n_components=n_comp)
        pca.fit(d)
        inAr = pca.explained_variance_ratio_
        outAr = [0] * 4
        outAr[dict[i]] = 1
        print inAr
        print outAr
        errors.append(network.back_propagate(inAr,outAr,.1))

    return errors

if __name__ == '__main__':
    print("Image -> Sift -> PCA -> NN -> Prediction!\n")

    n_comp = 5
    learning_rate = 0.1
    network = createNetwork(5, 5, 4)

    namesDict = exctractImgNames()
    names = namesDict.keys()
    shuffle(names)

    errors = trainNet(names,namesDict,network,learning_rate,n_comp)
    print errors

