import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray


images = []
img_original = ski.io.imread("images/calle.png")


img_gray = rgb2gray(img_original)
img_real_gray = ski.util.img_as_float(img_gray) * 255
images.append(img_real_gray)

img_clara = np.sqrt(255 * img_real_gray)
img_clara = ski.util.img_as_ubyte(img_clara / 255)
images.append(img_clara)

img_clara_cub = np.cbrt(255**2 * img_real_gray)
img_clara_cub = ski.util.img_as_ubyte(img_clara_cub / 255)
images.append(img_clara_cub)

img_eq = ski.exposure.equalize_hist(img_real_gray/255)
img_eq = ski.util.img_as_ubyte(img_eq)
images.append(img_eq)

img_eq_adapt_2 = ski.exposure.equalize_adapthist(img_real_gray/255, 2)
img_eq_adapt_2 = ski.util.img_as_ubyte(img_eq_adapt_2)
images.append(img_eq_adapt_2)

img_eq_adapt_4 = ski.exposure.equalize_adapthist(img_real_gray/255, 4)
img_eq_adapt_4 = ski.util.img_as_ubyte(img_eq_adapt_4)
images.append(img_eq_adapt_4)

img_eq_adapt_8 = ski.exposure.equalize_adapthist(img_real_gray/255, 8)
img_eq_adapt_8 = ski.util.img_as_ubyte(img_eq_adapt_8)
images.append(img_eq_adapt_8)

fig, axs = plt.subplots(7, 2, layout="constrained")

for i,image in enumerate(images):
    axs[i, 0].imshow(image, cmap=plt.cm.gray)
    h,c = ski.exposure.histogram(image)
    axs[i, 1].bar(c, h, 1.1)

axs_lineal = axs.ravel()
for i in range(0, axs_lineal.size, 2):
        axs_lineal[i].set_axis_off()
        axs_lineal[i + 1].set_xticks([0, 64, 128, 192, 255])
plt.show()
