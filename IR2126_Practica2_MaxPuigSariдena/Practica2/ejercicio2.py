import skimage as ski
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np


fig, axs = plt.subplots(7, 2, layout="constrained")
id_images = ["img_original","img_clara", "img_clara_cub","img_eq","img_eq_adapt_2","img_eq_adapt_4","img_eq_adapt_8"]

for i,id_image in enumerate(id_images):
    name = f'images/{id_image}.png'
    image = ski.io.imread(name)
    grey_im = rgb2gray(image)
    grey_im = ski.util.img_as_ubyte(grey_im)
    h, c = ski.exposure.histogram(grey_im)

    axs[i, 0].imshow(grey_im, cmap=plt.cm.gray)
    axs[i, 1].bar(c, h, 1.1)


axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 2):
        axs_lineal[i].set_axis_off()
        axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])
plt.show()