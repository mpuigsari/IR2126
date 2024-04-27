import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

path_imagen1 = "images/banderaItalia.jpg"
path_imagen2 = "images/banderaIrlanda.jpg"

imagen1 = ski.io.imread(path_imagen1)
imagen2 = ski.io.imread(path_imagen2)

if(imagen1.size > imagen2.size):
    imagenItalia = imagen2
    imagenIrlanda = imagen1
elif(imagen1.size < imagen2.size):
    imagenItalia = imagen1
    imagenIrlanda = imagen2

nueva = imagenIrlanda.copy()
imagenItalia = np.rot90(imagenItalia)
x_origin = int((imagenIrlanda.shape[0] - imagenItalia.shape[0]) / 2)
y_origin = int((imagenIrlanda.shape[1] - imagenItalia.shape[1]) / 2)
nueva[x_origin:imagenItalia.shape[0] + x_origin, y_origin:imagenItalia.shape[1] + y_origin, :] = imagenItalia


plt.imshow(nueva)
plt.show()