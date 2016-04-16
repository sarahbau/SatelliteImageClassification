from PIL import Image
from pylab import *
import sift.vlfeat as vlfeat
from sklearn.decomposition import PCA

if __name__ == '__main__':
    print("This is just a test!\n")

    imgName1 = '../images/Classified/Nature/slice_19-26.png'
    imgName2 = '../images/Classified/Nature/slice_19-27.png'

    # this image takes forever to compute - try way smaller please

    vlfeat.process_image(imgName1, 'tmp.sift1')
    l, d = vlfeat.read_features_from_file('tmp.sift1')

    vlfeat.process_image(imgName2, 'tmp.sift2')
    l2, d2 = vlfeat.read_features_from_file('tmp.sift2')

    # im = array(Image.open(imgName))
    # figure()
    # vlfeat.plot_features(im, l, True)
    # show()

    pca = PCA(n_components=3)
    pca.fit(d)
    print(pca.explained_variance_ratio_)

    pca = PCA(n_components=3)
    pca.fit(d2)
    print(pca.explained_variance_ratio_)
