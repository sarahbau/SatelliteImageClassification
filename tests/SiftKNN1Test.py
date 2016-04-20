from PIL import Image
from pylab import *
import numpy
import os
import sift.vlfeat as vlfeat
from sklearn.decomposition import PCA
from functools import total_ordering
from Queue import PriorityQueue
import pickle
import pprint
import csv
import copy


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
    #print "{}\t{}".format(v1, v2)
    return numpy.linalg.norm(v1-v2)

def split_list(a_list):
    half = len(a_list)/2
    return a_list[:half], a_list[half:]

def get_class(val, k, nature, commercial, residential=None):
    if val == None:
        #print "Why is this none?"
        return
    closest = None
    closest_dist = 100
    neighbors = PriorityQueue()
    for n in nature:
        #print n
        if n is None or n.shape != val.shape:
            continue
        neighbors.put(Neighbor(dist(n, val), 'nature', n))
    for n in commercial:
        if n is None or n.shape != val.shape:
            continue
        neighbors.put(Neighbor(dist(n, val), 'commercial', n))
    for n in residential:
        if n is None or n.shape != val.shape:
            continue
        neighbors.put(Neighbor(dist(n, val), 'residential', n))

    matches = {'nature': 0, 'commercial': 0, 'residential': 0}
    for i in xrange(k):
        if neighbors.empty():
            break
        neighbor = neighbors.get()
        #print 1/neighbor.dist
        matches[neighbor.t_class] += (1/neighbor.dist)
        #matches[neighbor.t_class] += 1

    return max(matches, key=matches.get)

def merge_dicts(x, *dicts):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = copy.deepcopy(x)
    for y in dicts:
        z['residential'].extend(y['residential'])
        z['commercial'].extend(y['commercial'])
        z['nature'].extend(y['nature'])
    # z.update(y)
    return z

if __name__ == '__main__':

    # nature_path = '../images/Classified/6/Nature/'
    # commercial_path = '../images/Classified/6/Commercial/'
    # residential_path = '../images/Classified/6/Residential/'
    # n_components = 10
    # nature = []
    # commercial = []
    # residential = []
    # for i in os.listdir(nature_path):
    #     nature.append(sift_image(nature_path + i, n_components))

    # for i in os.listdir(commercial_path):
    #     commercial.append(sift_image(commercial_path + i, n_components))

    # for i in os.listdir(residential_path):
    #    residential.append(sift_image(residential_path + i, n_components))

    # nature_test, nature_train = split_list(nature)
    # commercial_test, commercial_train = split_list(commercial)
    # residential_test, residential_train = split_list(residential)

    accuracy = {}
    for sift_comp in xrange(128,129):
        print "Starting sift", sift_comp
        accuracy[sift_comp] = {}
        data1 = pickle.load(open('../data/image1data_' + str(sift_comp) + 'c.dat', "rb"))['all']
        data2 = pickle.load(open('../data/image2data_' + str(sift_comp) + 'c.dat', "rb"))['all']
        data3 = pickle.load(open('../data/image3data_' + str(sift_comp) + 'c.dat', "rb"))['all']
        data4 = pickle.load(open('../data/image4data_' + str(sift_comp) + 'c.dat', "rb"))['all']
        #data5 = pickle.load(open('../data/image5data_' + str(sift_comp) + 'c.dat', "rb"))['all']
        data6 = pickle.load(open('../data/image6data_' + str(sift_comp) + 'c.dat', "rb"))['all']
        dataM = merge_dicts(data1, data2, data4, data4)

        # print "Length of data1: {}".format(len(data1['commercial']) + len(data1['residential']) + len(data1['nature']))
        # print "Length of data2: {}".format(len(data2['commercial']) + len(data2['residential']) + len(data2['nature']))
        # print "Length of data4: {}".format(len(data4['commercial']) + len(data4['residential']) + len(data4['nature']))
        # print "Length of data6: {}".format(len(data6['commercial']) + len(data6['residential']) + len(data6['nature']))
        # print "Length of dataM: {}".format(len(dataM['commercial']) + len(dataM['residential']) + len(dataM['nature']))
        #continue

        nature_train = dataM['nature']
        commercial_train = dataM['commercial']
        residential_train = dataM['residential']
        nature_test = data6['nature']
        commercial_test = data6['commercial']
        residential_test = data6['residential']

        # print "Nature Train: {}".format(len(nature_train))
        # print "Commercial Train: {}".format(len(commercial_train))
        # print "Residential Train: {}".format(len(residential_train))
        # print "Nature Test: {}".format(len(nature_test))
        # print "Commercial Test: {}".format(len(commercial_test))
        # print "Residential Test: {}".format(len(residential_test))
        #continue
        for k in xrange(1,101):
            print "Starting k =", k
            accuracy[sift_comp][k] = {}
            nature_matches = {'nature': 0, 'commercial': 0, 'residential': 0}
            count = 0
            for n in nature_test:
                #print "Nature: {}".format(count)
                # print get_class(n, k, nature_train, commercial_train)
                match = get_class(n, k, nature_train, commercial_train, residential_train)
                if match is None:
                    #print "Is this always none?"
                    continue
                nature_matches[match] += 1
                count += 1
            if count == 0:
                count = 1
            acc = nature_matches['nature']/float(count)
            accuracy[sift_comp][k]['nature'] = acc
            accuracy[sift_comp][k]['nat_count'] = count

            print "Nature: {}\tAccuracy: {}".format(nature_matches, acc)

            commercial_matches = {'nature': 0, 'commercial': 0, 'residential': 0}
            count = 0
            for n in commercial_test:
                #print "Commercial: {}".format(count)
                # print get_class(n, k, nature_train, commercial_train)
                match = get_class(n, k, nature_train, commercial_train, residential_train)
                if match is None:
                    continue
                commercial_matches[match] += 1
                count += 1
            if count == 0:
                count = 1
            acc = commercial_matches['commercial']/float(count)
            accuracy[sift_comp][k]['commercial'] = acc
            accuracy[sift_comp][k]['com_count'] = count

            print "Commercial: {}\tAccuracy: {}".format(commercial_matches, acc)

            residential_matches = {'nature': 0, 'commercial': 0, 'residential': 0}
            count = 0
            for n in residential_test:
                #print "Residential: {}".format(count)
                # print get_class(n, k, nature_train, commercial_train)
                match = get_class(n, k, nature_train, commercial_train, residential_train)
                if match is None:
                    continue
                residential_matches[match] += 1
                count += 1
            if count == 0:
                count = 1
            acc = residential_matches['residential']/float(count)
            accuracy[sift_comp][k]['residential'] = acc
            accuracy[sift_comp][k]['res_count'] = count

            print "residential: {}\tAccuracy: {}".format(residential_matches, acc)
    pprint.pprint(accuracy)

    with open('knnresult2.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)
        for sift, sval in accuracy.iteritems():
            for k, kval in sval.iteritems():
                nc = kval['nat_count']
                rc = kval['res_count']
                cc = kval['com_count']
                spamwriter.writerow([sift, k, (kval['nature']*nc+kval['commercial']*cc+kval['residential']*rc)/(nc + rc + cc)])


    #print(numpy.linalg.norm(pca2.explained_variance_ratio_-pca1.explained_variance_ratio_))