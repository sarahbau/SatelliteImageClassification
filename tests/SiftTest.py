from PIL import Image
from pylab import *
import numpy
import sift.vlfeat as vlfeat
from sklearn.decomposition import PCA

if __name__ == '__main__':
    print("This is just a test!\n")

    imgName1 = '../images/Classified/Nature/slice_19-26.png'
    imgName2 = '../images/Classified/Nature/slice_19-27.png'
    #imgName2 = '../images/Classified/Commercial/slice_4-1.png'

    # this image takes forever to compute - try way smaller please

    vlfeat.process_image(imgName1, 'tmp.sift1')
    l, d = vlfeat.read_features_from_file('tmp.sift1')

    vlfeat.process_image(imgName2, 'tmp.sift2')
    l2, d2 = vlfeat.read_features_from_file('tmp.sift2')

    # im = array(Image.open(imgName))
    # figure()
    # vlfeat.plot_features(im, l, True)
    # show()

    pca1 = PCA(n_components=10)
    pca1.fit(d)
    print(pca1.explained_variance_ratio_)

    pca2 = PCA(n_components=10)
    pca2.fit(d2)
    print(numpy.linalg.norm(pca2.explained_variance_ratio_-pca1.explained_variance_ratio_))