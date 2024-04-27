import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

img_original = ski.io.imread("images/lena256.pgm")


ta = ski.transform.AffineTransform(scale=(0.5, 0.75))
img_1 = ski.transform.warp(img_original.copy(), ta.inverse)
mescalado = ta.params

ta = ski.transform.AffineTransform(translation=(64, 0))
img_2 = ski.transform.warp(img_1, ta.inverse)
mtrasladado = ta.params

ta = ski.transform.AffineTransform(rotation=np.radians(15))
img_3 = ski.transform.warp(img_2, ta.inverse)
mrotacion = ta.params

shear4 = np.radians(-10)
ta = ski.transform.AffineTransform(shear=(shear4, shear4))
img_4 = ski.transform.warp(img_3, ta.inverse)
minclinado = ta.params

ta5 = ski.transform.AffineTransform(scale=(0.5, 0.75),translation=(64, 0), rotation=np.radians(15), shear=(shear4, shear4))
img_5 = ski.transform.warp(img_original.copy(), ta5.inverse)

ta6_params = minclinado @ mrotacion @ mtrasladado @ mescalado
ta6 = ski.transform.AffineTransform(ta6_params)
img_6 = ski.transform.warp(img_original.copy(), ta6.inverse)


list_afft = [None,mescalado,mtrasladado,mrotacion,minclinado, ta5.params, ta6_params]
list_img = [img_original,img_1,img_2, img_3, img_4, img_5, img_6]
fig, axs = plt.subplots(7, 1, layout="constrained")

for i, im in enumerate(list_img):
    if(i!=0):
        print(f'Matriz de transformación imagen {i} :\n{list_afft[i]}')
    axs[i].imshow(list_img[i], cmap=plt.cm.gray)
    axs[i].set_axis_off()


plt.show()