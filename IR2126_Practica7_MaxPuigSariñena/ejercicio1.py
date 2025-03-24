"""
Para ello, debes umbralizar la imagen mediante algún método de cálculo
automático del umbral, utilizar operaciones de morfología matemática
para eliminar las regiones pequeñas (ruido), rellenar huecos,
eliminar regiones que no sean redondas, etc. Después,
basándote en las propiedades de las regiones, debes generar un resultado
que muestre en rojo el área aproximada de las monedas de un euro y
en verde, el área de las de 10 céntimos.
El procesamiento que apliques debe ser exactamente el mismo para las tres
 imágenes.

 Ayuda: En la tercera imagen te puede resultar difícil separar las 3
 monedas que se están tocando entre sí.
 Para solucionar este problema, te proponemos que hagas una erosión
 con un disco lo suficientemente grande como para que en las monedas
 pequeñas permanezca tan solo una mínima semilla central.
 Seguidamente, haz una dilatación con un disco de un tamaño
 ligeramente inferior al usado en la erosión. De este modo,
 las monedas recuperarán un tamaño similar al original, pero sin
 volver a conectarse entre sí.
"""
import skimage as ski
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np


image1 = ski.io.imread("images/monedas1.png")
image2 = ski.io.imread("images/monedas2.png")
image3 = ski.io.imread("images/monedas3.png")
images = [image1,image2,image3]

fig, axs = plt.subplots(3, 3, layout="constrained")

for i, image in enumerate(images):
    umbral_global = ski.filters.threshold_otsu(image)
    binaria_global = image > umbral_global

    binaria_global = ndimage.binary_fill_holes(binaria_global).astype(int)

    erosion = ski.morphology.disk(18)
    dilatacion = ski.morphology.disk(16)
    img_erosion = ski.morphology.binary_erosion(binaria_global, footprint=(erosion))
    img_dilatacion = ski.morphology.binary_dilation(img_erosion, footprint=(dilatacion))
    img_cierre = img_dilatacion
    img_etiquetada = ski.morphology.label(img_cierre)

    props = ski.measure.regionprops(img_etiquetada)

    img_monedas = np.zeros((256,256,3))
    for p in props:
        if p.eccentricity < 0.75:
            if p.area > 900:
                img_monedas[img_etiquetada == p.label] = [255,0,0]
            elif p.area < 900:
                img_monedas[img_etiquetada == p.label] = [0, 255, 0]


    axs[i, 0].imshow(binaria_global, cmap="gray")
    axs[i, 0].set_title(f'Umbral: {umbral_global}', fontsize=16)
    axs[i, 1].imshow(img_cierre, cmap="gray")
    axs[i,2].imshow(img_monedas.astype(np.uint8), cmap="gray")
for ax in axs.ravel():
    ax.set_axis_off()
plt.show()


