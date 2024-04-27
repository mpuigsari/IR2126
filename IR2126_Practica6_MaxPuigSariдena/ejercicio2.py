import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import scipy

image1 = ski.io.imread("images/cuadros.png")  # Probar también con sintética
image1 = ski.util.img_as_float(image1)
image1 = ski.util.random_noise(image1, mode="gaussian", var=0.001)

rot_list = [0, 22.5, 45, 67.5, 90]
rot_str_list = ["0º", "22.5º", "45º", "67.5º", "90º"]
images = []
for r in rot_list:
    images.append(scipy.ndimage.rotate(image1, angle=r))


def ajustar_01(imagen):
    maxi = imagen.max()
    mini = imagen.min()
    return (imagen - mini) / (maxi - mini)


def filtrar(image, nombres_filtros):
    images = [image]
    for nf in nombres_filtros:
        if nf == "moravec":
            img = my_moravec(image)
        else:
            if nf == "fast":
                param = ", 9"
            else:
                param = ""
            img = eval("ski.feature.corner_" + nf + "(image" + param + ")")
            if nf == "foerstner":
                img = img[0]
            elif nf == "kitchen_rosenfeld":
                img = np.abs(img)
        images.append(img)
    return images


def detectar_picos(images, umbral):
    resultados = [images[0]]
    for i in range(1, len(images)):
        img = ski.feature.corner_peaks(images[i], indices=False, min_distance=10, threshold_rel=umbral)
        resultados.append(img)
    return resultados


def my_moravec(cimage, window_size=1):
    rows = cimage.shape[0]
    cols = cimage.shape[1]
    out = np.zeros(cimage.shape)
    for r in range(2 * window_size, rows - 2 * window_size):
        for c in range(2 * window_size, cols - 2 * window_size):
            min_msum = 1E100
            for br in range(r - window_size, r + window_size + 1):
                for bc in range(c - window_size, c + window_size + 1):
                    if br != r or bc != c:  #### En scikit-image aquí aparece un AND !!!!
                        msum = 0
                        for mr in range(- window_size, window_size + 1):
                            for mc in range(- window_size, window_size + 1):
                                t = cimage[r + mr, c + mc] - cimage[br + mr, bc + mc]
                                msum += t * t
                        min_msum = min(msum, min_msum)

            out[r, c] = min_msum
    return out


def mostrar(titulo, resultados1, nombres):
    fig, ax = plt.subplots(nrows=1, ncols=len(resultados1), layout="constrained")
    fig.suptitle(titulo, fontsize=24)
    for i in range(len(resultados1)):
        ax[i].imshow(resultados1[i], cmap='gray')
        ax[i].set_title(nombres[i], fontsize=16)
    for a in ax.ravel():
        a.set_axis_off()
    plt.show()


filtros = ["kitchen_rosenfeld", "foerstner", "moravec", "harris", "fast"]

UMBRAL = 0.05
i=0
for image in images:
    images_i = filtrar(image,filtros)
    picos_i = detectar_picos(images_i, UMBRAL)
    mostrar(f"Imagen Rotada {rot_str_list[i]} (umbral = {UMBRAL})", picos_i, [rot_str_list[i]] + filtros)
    i+=1
#0.01 Fast ---0.05 Forestner ---0.075 Fast --- 0.1 fast ---