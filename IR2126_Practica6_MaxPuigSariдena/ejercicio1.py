import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

# Leer la imagen y crear mapa de bordes usando el gradiente de Sobel
imagen = ski.io.imread("images/cuadros.png")
str_imagenes = ["Original", "Ruido 0.001", "Ruido 0.0015", "Ruido 0.0025"]

gaussian_values = (0.001,  # sigma = 0.032
                   0.0015,  # sigma = 0.071
                   0.0025)  # sigma = 0.1
images = [imagen]
for i in range(len(gaussian_values)):
    img_noise = ski.util.random_noise(imagen, mode="gaussian", var=gaussian_values[i])
    images.append(img_noise)

def sobel(image):
    gradiente = ski.filters.sobel(image)

    maximo = gradiente.max()
    low = maximo * 0.1
    high = maximo * 0.2
    mapa_bordes = ski.filters.apply_hysteresis_threshold(gradiente, low, high)

    mapa_bordes = ski.morphology.thin(mapa_bordes)  # Reduce el grosor de los bordes a un solo píxel
    resultado1 = mapa_bordes

    # Transfromada de Hough Probabilística y Proresiva
    segmentos = ski.transform.probabilistic_hough_line(mapa_bordes, threshold=10, line_length=5, line_gap=3)
    resultado2 = segmentos
    return [resultado1, resultado2]


def canny(image):
    mapa_bordes = ski.feature.canny(image, sigma=3)
    resultado3 = mapa_bordes

    segmentos = ski.transform.probabilistic_hough_line(mapa_bordes, threshold=10, line_length=5, line_gap=3)
    resultado4 = segmentos

    return [resultado3, resultado4]


def mostrar(img_fila1, msg_fila1, titulo):
    fig, ax = plt.subplots(nrows=1, ncols=len(img_fila1), layout="constrained")
    fig.suptitle(titulo, fontsize=24)
    for i in range(len(img_fila1)):
        if(i%2==0):
            ax[i].imshow(img_fila1[i], cmap='gray')
        else:
            ax[i].imshow(np.zeros(img_fila1[0].shape), cmap='gray')
            for segmento in img_fila1[i]:
                p0, p1 = segmento
                ax[i].plot((p0[0], p1[0]), (p0[1], p1[1]), color='r')  # Dibujar segmento
        ax[i].set_title(msg_fila1[i], fontsize=16)
    for a in ax.ravel():
        a.set_axis_off()
    plt.show()


for i in range(len(images)):
    res1_2 = sobel(images[i])
    res3_4 = canny(images[i])
    lista_imagenes = [res1_2[0], res1_2[1], res3_4[0], res3_4[1]]
    for x in lista_imagenes:
        print(lista_imagenes)
    mostrar(lista_imagenes, msg_fila1=[str_imagenes[i], "Sobel 1px", "Hough", "Canny", "Hough"],titulo=str_imagenes[i])

