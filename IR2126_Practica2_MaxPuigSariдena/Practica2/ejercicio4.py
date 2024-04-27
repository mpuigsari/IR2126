import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from skimage.color.adapt_rgb import adapt_rgb, each_channel,hsv_value


def hist_rgb(axis, i, image):
    colors = ("red", "green", "blue")
    for channel_id, color in enumerate(colors):
        h, c = ski.exposure.histogram(image[:, :, channel_id])
        axis[i, channel_id+1].bar(c, h, 1.1)

@adapt_rgb(each_channel)
def eq_each_adapt(image):
    return ski.exposure.equalize_hist(image)

@adapt_rgb(hsv_value)
def eq_each_hsv(image):
    return ski.exposure.equalize_hist(image)


img_original = ski.io.imread("images/calle.png")
#h_orig, c_orig = ski.exposure.histogram(img_original)

img_real = ski.util.img_as_float(img_original)
img_eq_each = eq_each_adapt(img_real)
img_eq_each = ski.util.img_as_ubyte(img_eq_each)

img_eq_hsv = eq_each_hsv(img_real)
img_eq_hsv = ski.util.img_as_ubyte(img_eq_hsv)

images = [img_original, img_eq_each,img_eq_hsv]
fig, axs = plt.subplots(3, 4, layout="constrained")

for i,image in enumerate(images):
    axs[i, 0].imshow(image, cmap=plt.cm.gray)
    hist_rgb(axs,i,image)


axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 4):
        axs_lineal[i].set_axis_off()
        axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])
        axs_lineal[i + 2].set_xticks([0, 64, 128, 192, 255])
        axs_lineal[i + 3].set_xticks([0, 64, 128, 192, 255])
plt.show()
