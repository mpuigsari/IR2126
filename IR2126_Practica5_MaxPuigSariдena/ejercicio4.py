import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time
import scipy.fft as fft


MASK_SIZES = list(range(3, 22, 2))
STD_DEV = 3

imagen = ski.io.imread("images/boat.511.tiff")
imagen = ski.util.img_as_float(imagen)

timesEspacio, timesFrec = [], []
repeat = 10

for mask_size in MASK_SIZES:
    # Convolución en el espacio
    mascara = np.ones((mask_size, mask_size))  # Máscara de NxN toda con 1
    mascara /= np.sum(mascara)  # Máscara normalizada

    inicio_espacio = time.time()
    for mean in range(repeat):
        res_convol = scipy.ndimage.convolve(imagen, mascara, mode="wrap")
    fin_espacio = time.time()
    # Ampliamos la máscara con ceros para que tenga el mismo tamaño que la imagen
    mascara_centrada = np.zeros(imagen.shape)
    fila_i = imagen.shape[0] // 2 - mask_size // 2
    col_i = imagen.shape[1] // 2 - mask_size // 2
    mascara_centrada[fila_i:fila_i + mask_size, col_i:col_i + mask_size] = mascara

    # Pasamos imagen y máscara a las frecuencias
    FTimagen = fft.fft2(imagen)
    mascara_en_origen = fft.ifftshift(mascara_centrada)
    FTmascara = fft.fft2(mascara_en_origen)

    inicio_frec = time.time()
    for mean in range(repeat):
        # Convolución en la frecuancia
        FTimagen_filtrada = FTimagen * FTmascara  # Producto punto a punto
    fin_frec = time.time()

    timesEspacio.append(float((fin_espacio - inicio_espacio) / repeat))
    timesFrec.append(float((fin_frec - inicio_frec) / repeat))
timeEspacio = np.mean(timesEspacio)
timeFrec = np.mean(timesFrec)

print(f"Tiempo empleado en Frecuencia: {timeFrec:0.9f}")
print(f"Tiempo empleado en Espacio: {timeEspacio:0.9f}")

fig, axs = plt.subplots(1, 3, layout="constrained")
axs[0].imshow(imagen, cmap=plt.cm.gray)
axs[0].axis('off')
axs[1].plot(MASK_SIZES, timesEspacio, 'r-', label='2D')
axs[1].plot(MASK_SIZES, timesFrec, 'b-', label='1D')
axs[2].axis('off')
plt.show()
