import skimage as ski
import matplotlib.pyplot as plt

alpha = 1.0

I = ski.io.imread("images/borrosa.png")
I = ski.util.img_as_float(I)

F = ski.filters.gaussian(I, sigma=3)
R = I + alpha * (I - F)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(I, cmap=plt.cm.gray)
axs[0].axis('off')

axs[1].imshow(R, cmap=plt.cm.gray)
axs[1].axis('off')

axs[2].imshow(F, cmap=plt.cm.gray)
axs[2].axis('off')

plt.show()