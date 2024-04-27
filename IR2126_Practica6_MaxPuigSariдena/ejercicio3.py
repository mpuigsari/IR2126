import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

# Leer la imagen y crear mapa de bordes usando el gradiente de Sobel
image1 = ski.io.imread("images/monedas1.png")
#image1 = ski.color.rgb2gray(image1)  # Convertir a niveles de gris
image2 = ski.io.imread("images/monedas2.png")
#image2 = ski.color.rgb2gray(image2)  # Convertir a niveles de gris
image3 = ski.io.imread("images/monedas3.png")
#image3 = ski.color.rgb2gray(image3)  # Convertir a niveles de gris

images = [image1,image2,image3]


def monedas(image):
    gradiente = ski.filters.sobel(image)
    maximo = gradiente.max()
    low = maximo * 0.28
    high = maximo * 0.32
    mapa_bordes = ski.filters.apply_hysteresis_threshold(gradiente, low, high)
    mapa_bordes = ski.morphology.thin(mapa_bordes)  # Reduce el grosor de los bordes a un solo píxel
    # Transformada de Hough para círculos

    resultado = np.zeros((256, 256, 3))
    radios_posibles = np.arange(18, 27, 1)  # Buscará cícculos con radios entre 10 y 30 de 2 en 2
    hough_res = ski.transform.hough_circle(mapa_bordes, radios_posibles)
    accums, cx, cy, radii = ski.transform.hough_circle_peaks(hough_res, radios_posibles, min_xdistance=10, min_ydistance=10,
                                                             threshold=hough_res.max() / 2)  # El threshold no debería ser necesario. Se supone que es el valor por defecto
    for fila, col, radio in zip(cy, cx, radii):
        if radio > 20:
            i=0
        else:
            i=1
        circy, circx = ski.draw.circle_perimeter(fila, col, radio, shape=image.shape)  # Dibuja un círculo
        resultado[circy, circx, i] = 1
    return resultado

fig, axs = plt.subplots(nrows=1, ncols=3)
fig.suptitle("Monedas con colorines", fontsize=24)

for i in range(len(images)):
    axs[i].imshow(monedas(images[i]), cmap='gray')
    axs[i].set_title(f'Imagen Monedas {i+1}', size=16)
for ax in axs:
    ax.set_axis_off()

plt.show()
