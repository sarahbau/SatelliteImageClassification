from __future__ import division
from PIL import Image
import math
import os


def long_slice(image_path, out_name, outdir, slice_size):
    """slice an image into parts slice_size tall"""
    img = Image.open(image_path)
    width, height = img.size
    upper = 0
    left = 0
    v_slices = int(math.ceil(height/slice_size))
    h_slices = int(math.ceil(width/slice_size))

    vcount = 1
    for vslice in xrange(v_slices):
        if vcount == v_slices:
            lower = height
        else:
            lower = int(vcount * slice_size)
        hcount = 1
        left = 0
        for hslice in xrange(h_slices):
            #if we are at the end, set the lower bound to be the bottom of the image
            if hcount == h_slices:
                right = width
            else:
                right = int(hcount * slice_size)

            #set the bounding box! The important bit
            bbox = (left, upper, right, lower)
            working_slice = img.crop(bbox)

            left += slice_size
            #save the slice
            working_slice.save(os.path.join(outdir, "slice_" + str(vcount) + "-" + str(hcount) + ".png"))
            hcount += 1
        upper += slice_size
        vcount += 1

if __name__ == '__main__':
    #slice_size is the max height of the slices in pixels
    for file in os.listdir(os.getcwd()):
        if file.endswith(".png"):
            d = file.replace(".png", "")
            path = os.getcwd() + "/" + d
            os.makedirs(path)
            long_slice(file, "1", path, 200)
