import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from skimage.color.adapt_rgb import adapt_rgb, each_channel



@adapt_rgb(each_channel)
def eq_each(image):
    return ski.exposure.equalize_hist(image)


@adapt_rgb(each_channel)
def eqadapt_each(image,k):
    return ski.exposure.equalize_adapthist(image, kernel_size=k)

def hist_rgb(axis, i, image):
    colors = ("red", "green", "blue")
    for channel_id, color in enumerate(colors):
        h, c = ski.exposure.histogram(image[:, :, channel_id])
        axis[i, channel_id+1].bar(c, h, 1.1)


img_original = ski.io.imread("images/calle.png")
#h_orig, c_orig = ski.exposure.histogram(img_original)


img_real = ski.util.img_as_float(img_original) * 255

img_clara = np.sqrt(255 * img_real)
img_clara = ski.util.img_as_ubyte(img_clara / 255)
#h_clara, c_clara = ski.exposure.histogram(img_clara)

img_clara_cub = np.cbrt(255**2 * img_real)
img_clara_cub = ski.util.img_as_ubyte(img_clara_cub / 255)
#h_clara_cub, c_clara_cub = ski.exposure.histogram(img_clara_cub)


img_eq = eq_each(img_real/255)
img_eq = ski.util.img_as_ubyte(img_eq)
#h_eq, c_eq = ski.exposure.histogram(img_eq)

img_eq_adapt_2 = eqadapt_each(img_real/255, 2)
img_eq_adapt_2 = ski.util.img_as_ubyte(img_eq_adapt_2)
#h_eq_adapt_2, c_eq_adapt_2 = ski.exposure.histogram(img_eq_adapt_2)

img_eq_adapt_4 = eqadapt_each(img_real/255, 4)
img_eq_adapt_4 = ski.util.img_as_ubyte(img_eq_adapt_4)
#h_eq_adapt_4, c_eq_adapt_4 = ski.exposure.histogram(img_eq_adapt_4)

img_eq_adapt_8 = eqadapt_each(img_real/255, 8)
img_eq_adapt_8 = ski.util.img_as_ubyte(img_eq_adapt_8)
#h_eq_adapt_8, c_eq_adapt_8 = ski.exposure.histogram(img_eq_adapt_8)

fig, axs = plt.subplots(7, 4, layout="constrained")
images = [img_original,img_clara, img_clara_cub,img_eq,img_eq_adapt_2,img_eq_adapt_4,img_eq_adapt_8]
id_images = ["img_original","img_clara", "img_clara_cub","img_eq","img_eq_adapt_2","img_eq_adapt_4","img_eq_adapt_8"]


for i,image in enumerate(images):
    axs[i, 0].imshow(image, cmap=plt.cm.gray)
    hist_rgb(axs,i,image)
    name = f'images/{id_images[i]}.png'
    ski.io.imsave( fname=name, arr=image)

axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 4):
        axs_lineal[i].set_axis_off()
        axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])
        axs_lineal[i + 2].set_xticks([0, 64, 128, 192, 255])
        axs_lineal[i + 3].set_xticks([0, 64, 128, 192, 255])
plt.show()
