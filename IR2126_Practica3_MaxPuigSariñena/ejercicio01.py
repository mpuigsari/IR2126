import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

img_original = ski.io.imread("images/lena256.pgm")

angle = 30

img_girada = ski.transform.rotate(img_original, angle)
angle_euc = np.radians(360-angle)
translation = np.array(img_original.shape)/2
tf_euclidea = ski.transform.EuclideanTransform(translation=(-translation))
tf_euclidea_1 = ski.transform.EuclideanTransform(rotation=angle_euc)
tf_euclidea_2 = ski.transform.EuclideanTransform(translation=translation)

tf_euclidea_p = tf_euclidea_2.params @ tf_euclidea_1.params @ tf_euclidea.params

tf_euclidea = ski.transform.EuclideanTransform(tf_euclidea_p)

img_girada_euclidea = ski.transform.warp(img_original, tf_euclidea.inverse, order=3)

fig, axs = plt.subplots(1, 3, layout="constrained")
axs[0].imshow(img_original, cmap=plt.cm.gray)
axs[1].imshow(img_girada, cmap=plt.cm.gray)
axs[2].imshow(img_girada_euclidea, cmap=plt.cm.gray)
plt.show()
