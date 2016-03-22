from PIL import Image


image_path = 'C:\Users\Nathan\Desktop\m_4308960_ne_16_1_20100702.png'
out_path = 'C:\Users\Nathan\Desktop\out.png'

image = Image.open(image_path)
image_pix = image.load()

for x in xrange(0, image.size[0]):
    if x % 100 == 0:
        print (x / (1.0 * image.size[0]))
    for y in xrange(0, image.size[1]):
        mult = image_pix[x, y][3]/255.0
        image_pix[x, y] = (
            int(image_pix[x, y][0] * mult),
            int(image_pix[x, y][1] * mult),
            int(image_pix[x, y][2] * mult),
            255
        )

image.save(out_path)
