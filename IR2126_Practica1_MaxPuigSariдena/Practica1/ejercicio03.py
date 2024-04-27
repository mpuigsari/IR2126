import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

def errorCuadráticoMedioBanda(m1, m2):  # Las bandas dadas deben ser flotantes
    mo1 = m1 * 255  # Queremos medir el error en el rango [0,255]
    mo2 = m2 * 255
    return np.sum(np.power(mo2 - mo1, 2), None) / m1.size


def errorMedioBanda(m1, m2):  # Las bandas dadas deben ser flotantes
    mo1 = m1 * 255  # Queremos medir el error en el rango [0,255]
    mo2 = m2 * 255
    return np.sum(abs(mo2 - mo1), None) / m1.size


def maximoErrorAbsolutoBanda(m1, m2):  # Las bandas dadas deben ser flotantes
    mo1 = m1 * 255  # Queremos medir el error en el rango [0,255]
    mo2 = m2 * 255
    return np.max(np.abs(mo2 - mo1))


imagenOriginal = ski.io.imread("images/mapas.png")
ski.io.imsave("images/mapas100.jpg", arr=imagenOriginal, quality=100)
ski.io.imsave("images/mapas75.jpg", arr=imagenOriginal, quality=75)
ski.io.imsave("images/mapas15.jpg", arr=imagenOriginal, quality=15)
imagenOriginal = ski.util.img_as_float(imagenOriginal)  # Valores flotantes en el rango [0,1]


im_100 = ski.io.imread("images/mapas100.jpg")
im_75 = ski.io.imread("images/mapas75.jpg")
im_15 = ski.io.imread("images/mapas15.jpg")

guardadas = [im_100, im_75, im_15]
guardadas_name = ["Imagen 100%","Imagen 75%","Imagen 15%"]
bandas = ["Roja", "Verde", "Azul"]

fig, axs = plt.subplots(3, 3, layout="constrained")
for indice, imagen in enumerate(guardadas):
    imagenGuardada = ski.util.img_as_float(imagen)
    print(f"\n\nImagen {guardadas_name[indice]}:\n")
    for nBanda, nombre in enumerate(bandas):
        print(f"Banda {nombre}")
        print(f"   Máximo error: {maximoErrorAbsolutoBanda(imagenOriginal[:, :, nBanda], imagenGuardada[:, :, nBanda])}")
        print(f"   Error medio: {errorMedioBanda(imagenOriginal[:, :, nBanda], imagenGuardada[:, :, nBanda])}")
        print(f"   Error cuadrático medio: {errorCuadráticoMedioBanda(imagenOriginal[:, :, nBanda], imagenGuardada[:, :, nBanda])}")

    errores = []
    for nBanda in range(3):
        errorBanda = abs(imagenGuardada[:, :, nBanda] - imagenOriginal[:, :, nBanda])
        errores.append(errorBanda)
    errorGlobal = np.stack(errores, axis=-1)
    axs[indice][0].imshow(imagenOriginal)
    axs[indice][1].imshow(imagenGuardada)
    axs[indice][2].imshow(errorGlobal/errorGlobal.max())  # Piensa cómo mejorar la visualización del error


for ax in axs.ravel():
    ax.set_axis_off()
plt.show()