import sift.vlfeat as sift
from sklearn.decomposition import PCA
import os
from random import shuffle
import layer.Network as Network
import layer.Layer as Layer
import pickle
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
    l1 = Layer.Layer(in_neurons, hidden_layer_count)
    l2 = Layer.Layer(hidden_layer_count, n_outs)
    l1.randomize(-1, 1)
    l2.randomize(-1, 1)
    network = Network.Network()
    network.add_layer(l1)
    network.add_layer(l2)
    return network

def trainNet(file_names,dict,network,rate,n_comp):
    errors = []
    for i in file_names:
        os.remove('tmp.sift')
        os.remove('tmp.pgm')
        try:
            sift.process_image(i, 'tmp.sift')
            l, d = sift.read_features_from_file('tmp.sift')
            pca = PCA(n_components=n_comp)
            pca.fit(d)
            inAr = pca.explained_variance_ratio_.tolist()
            outAr = [0] * 4
            outAr[dict[i]] = 1
            errors.append(network.back_propagate(inAr,outAr,rate))
        except IndexError:
            print("File "+i+" is broken!\n")
            continue
        except ValueError:
            print("Woops value error!\n")
            continue
    return errors

def toidx(ar):
    out = 0;
    max = -100.0;
    for i in range(len(ar)):
        if ar[i] > max:
            out = i
            max = ar[i]
    return out

def test(file_names,dict,network):
    errors = []
    for i in file_names:
        try:
            sift.process_image(i, 'tmp.sift')
            l, d = sift.read_features_from_file('tmp.sift')
            pca = PCA(n_components=5)
            pca.fit(d)
            inAr = pca.explained_variance_ratio_.tolist()
            outAr = [0] * 4
            outAr[dict[i]] = 1
            actual = network.get_output(inAr)
            print toidx(actual)
            if dict[i] == toidx(actual):
                errors.append(1)
            else:
                errors.append(0)
        except IndexError:
            print("File " + i + " is broken!\n")
            continue
        except ValueError:
            print("Woops value error!\n")
            continue
    return errors


if __name__ == '__main__':
    print("Image -> Sift -> PCA -> NN -> Prediction!\n")

    #n_comp = 20
    #learning_rate = 0.2
    #network = createNetwork(20, 100, 4)

    namesDict = exctractImgNames()
    names = namesDict.keys()
    shuffle(names)

    #pickle.dump(network, open("sift_20_100_4", "wb"))
    #trainNet(names,namesDict,network,learning_rate,n_comp)

    network = pickle.load(open("sift_5_5_4","r"))
    errors = test(names,namesDict,network)

    tot = sum(errors) / len(errors)

    print tot

