import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

img_original = ski.io.imread("images/Albert_Einstein.jpg")
print(img_original.shape)
src = [[0, 0],
       [0, 1066],
       [310, 340],  # ceja izquierdo
       [540, 360],  # ceja  derecho
       [280, 240],  # frente izq
       [580, 220],  # frente der
       [440, 195],  # frente alta
       [65, 270],  # pelo izq
       [730, 275],  # pelo der
       [430, 60],  # pelo alta
       [490, 520],  # nariz
       [460, 340],  # entrecejo
       [400, 610],  # labios izqda
       [510, 600],  # labios dcha
       [460, 710],  # barbilla
       [799, 0],
       [799, 1066]]

dst = src
src = np.array(src)
dst = np.array(dst)

dst[4] = [290, 310]
dst[5] = [580, 300]
dst[6] = [480, 280]


tform = ski.transform.PiecewiseAffineTransform()
tform.estimate(src, dst)
img_t = ski.transform.warp(img_original, inverse_map=tform.inverse)

fig, axs = plt.subplots(1, 2, layout="constrained")
axs[0].imshow(img_original, cmap=plt.cm.gray)
axs[0].plot(src[:, 0], src[:, 1], '.r')
axs[1].imshow(img_t, cmap=plt.cm.gray)

axs[0].set_axis_off()
axs[1].set_axis_off()
plt.show()