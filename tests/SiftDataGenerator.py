from PIL import Image
from pylab import *
import numpy
import os
import time
import pickle
import sift.vlfeat as vlfeat
from sklearn.decomposition import PCA

def sift_image(path, n_components):
    if not path.endswith(".png"): return
    vlfeat.process_image(path, 'tmp.sift1')
    #print "File is {} bytes".format(os.stat('tmp.sift1').st_size)
    if(os.stat('tmp.sift1').st_size < 378):
        print "Bad tmp.sift1 for", path
        return
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

    #image_num = 2
    #n_components = 10
    for image_num in xrange(2, 7):
        for n_components in xrange(2, 21):
            nature_path = '../images/Classified/' + str(image_num) + '/Nature/'
            commercial_path = '../images/Classified/' + str(image_num) + '/Commercial/'
            residential_path = '../images/Classified/' + str(image_num) + '/Residential/'

            save_name = "../data/image" + str(image_num) + "data"

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
            data['all'] = {
                'nature': nature,
                'commercial': commercial,
                'residential': residential
            }
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

            pickle.dump(data, open(save_name, "wb"))
