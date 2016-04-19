from PIL import Image
from pylab import *
import numpy
import os
import pickle
import sift.vlfeat as vlfeat
from sklearn.decomposition import PCA

def sift_image(path, n_components):
    vlfeat.process_image(path, 'tmp.sift1')
    l, d = vlfeat.read_features_from_file('tmp.sift1')
    pca1 = PCA(n_components=n_components)
    pca1.fit(d)
    return pca1.explained_variance_ratio_

def dist(v1, v2):
    return numpy.linalg.norm(v1-v2)

def split_list(a_list):
    half = len(a_list)/2
    return a_list[:half], a_list[half:]

if __name__ == '__main__':

    nature_path = '../images/Classified/6/Nature/'
    commercial_path = '../images/Classified/6/Commercial/'
    residential_path = '../images/Classified/6/Residential/'
    n_components = 10
    save_name = "../data/image1data"


    save_name = save_name + '_' + str(n_components) + 'c' + '.dat'

    nature = []
    commercial = []
    residential = []
    for i in os.listdir(nature_path):
        nature.append(sift_image(nature_path + i, n_components))

    for i in os.listdir(commercial_path):
        commercial.append(sift_image(commercial_path + i, n_components))

    for i in os.listdir(residential_path):
        residential.append(sift_image(residential_path + i, n_components))

    nature_test, nature_train = split_list(nature)
    commercial_test, commercial_train = split_list(commercial)
    residential_test, residential_train = split_list(residential)

    data = dict()
    data['test'] = {
        'nature': nature_test,
        'commercial': commercial_test,
        'residential': residential_test
    }
    data['train'] = {
        'nature': nature_train,
        'commercial': commercial_train,
        'residential': residential_train
    }

    pickle.dump( data, open( save_name, "wb"))