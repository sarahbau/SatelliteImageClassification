from PIL import Image
from pylab import *
import sift.vlfeat as vlfeat
from sklearn.decomposition import PCA

if __name__ == '__main__':
    print("This is just a test!\n")

    imgName = '../images/raleigh/small_image.png'

    # this image takes forever to compute - try way smaller please

    vlfeat.process_image(imgName, 'tmp.sift')
    l, d = vlfeat.read_features_from_file('tmp.sift')

    # im = array(Image.open(imgName))
    # figure()
    # vlfeat.plot_features(im, l, True)
    # show()

    print d

    pca = PCA(n_components=2)
    pca.fit(d)
    print(pca.explained_variance_ratio_)
