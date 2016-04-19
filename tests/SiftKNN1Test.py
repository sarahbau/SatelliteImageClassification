from PIL import Image
from pylab import *
import numpy
import os
import sift.vlfeat as vlfeat
from sklearn.decomposition import PCA
from functools import total_ordering
from Queue import PriorityQueue


@total_ordering
class Neighbor(object):

    def __init__(self, dist, t_class, attr):
        #print "Adding neighbor of class {} and dist {}".format(t_class, dist)
        self.dist = dist
        self.t_class = t_class
        self.attr = attr

    def __lt__(self, other):
        return self.dist < other.dist

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

def get_class(val, k, nature, commercial, residential=None):
    closest = None
    closest_dist = 100
    neighbors = PriorityQueue()
    for n in nature:
        neighbors.put(Neighbor(dist(n, val), 'nature', n))
    for n in commercial:
        neighbors.put(Neighbor(dist(n, val), 'commercial', n))
    for n in residential:
        neighbors.put(Neighbor(dist(n, val), 'residential', n))

    matches = {'nature': 0, 'commercial': 0, 'residential': 0}
    for i in xrange(k):
        matches[neighbors.get().t_class] += 1

    return max(matches, key=matches.get)

if __name__ == '__main__':

    nature_path = '../images/Classified/6/Nature/'
    commercial_path = '../images/Classified/6/Commercial/'
    residential_path = '../images/Classified/6/Residential/'
    n_components = 10
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

    all_list = nature_train + commercial_train + residential_train

    k = 5
    nature_matches = {'nature': 0, 'commercial': 0, 'residential': 0}
    for n in nature_test:
        # print get_class(n, k, nature_train, commercial_train)
        nature_matches[get_class(n, k, nature_train, commercial_train, residential_train)] += 1

    print nature_matches
    print "switch"

    commercial_matches = {'nature': 0, 'commercial': 0, 'residential': 0}
    for n in commercial_test:
        # print get_class(n, k, nature_train, commercial_train)
        commercial_matches[get_class(n, k, nature_train, commercial_train, residential_train)] += 1

    print commercial_matches

    residential_matches = {'nature': 0, 'commercial': 0, 'residential': 0}
    for n in residential_test:
        # print get_class(n, k, nature_train, commercial_train)
        residential_matches[get_class(n, k, nature_train, commercial_train, residential_train)] += 1

    print residential_matches



    #print(numpy.linalg.norm(pca2.explained_variance_ratio_-pca1.explained_variance_ratio_))