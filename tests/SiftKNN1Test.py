from PIL import Image
from pylab import *
import numpy
import os
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

def get_class(val, list_a, list_b):
    closest = None
    closest_dist = 100
    for a in list_a:
        d = dist(a, val)
        if (d < closest_dist):
            closest = a
            closest_dist = d
    inB = False
    for b in list_b:
        d = dist(b, val)
        if (d < closest_dist):
            inB = True
            closest = b
            closest_dist = d

    if (inB):
        return 1
    else:
        return 0

if __name__ == '__main__':

    nature_path = '../images/Classified/Nature/'
    commercial_path = '../images/Classified/Commercial/'
    residential_path = '../images/Classified/Residential/'
    n_components = 5
    nature = []
    commercial = []
    residential = []
    for i in os.listdir(nature_path):
        nature.append(sift_image(nature_path + i, n_components))

    for i in os.listdir(commercial_path):
        commercial.append(sift_image(commercial_path + i, n_components))

    #for i in os.listdir(residential_path):
    #    residential.append(sift_image(residential_path + i, n_components))

    nature_test, nature_train = split_list(nature)
    commercial_test, commercial_train = split_list(commercial)

    all_list = nature_train + commercial_train
    
    for n in nature_test:
        print (get_class(n, nature_train, commercial_train))

    print "switch"

    for n in commercial_test:
        print (get_class(n, nature_train, commercial_train))


    #print(numpy.linalg.norm(pca2.explained_variance_ratio_-pca1.explained_variance_ratio_))