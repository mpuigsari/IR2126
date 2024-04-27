import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time

MASK_SIZES = list(range(3, 22, 2))
STD_DEV = 3

imagen = ski.io.imread("images/boat.512.tiff")
imagen = ski.util.img_as_float(imagen)

times2D, times1D = [], []
repeat = 10

for mask_size in MASK_SIZES:
    vector = scipy.signal.windows.gaussian(mask_size, STD_DEV)
    vector /= np.sum(vector)

    vectorH = vector.reshape(1, mask_size)
    vectorV = vector.reshape(mask_size, 1)

    matriz = vectorV @ vectorH

    inicio2D = time.time()
    for mean in range(repeat):
        res_convol2D = scipy.ndimage.convolve(imagen, matriz)
    fin2D = time.time()

    inicio1D = time.time()
    for mean in range(repeat):
        resH = scipy.ndimage.convolve(imagen, vectorH)
        res1D = scipy.ndimage.convolve(resH, vectorV)
    fin1D = time.time()

    times2D.append(float((fin2D - inicio2D) / repeat))
    times1D.append(float((fin1D - inicio1D) / repeat))
time1D = np.mean(times1D)
time2D = np.mean(times2D)
print("¿Obtenemos el mismo resultado?", np.allclose(res_convol2D, res1D))
print(f"Tiempo empleado con máscara  2D: {time2D:0.9f}")
print(f"Tiempo empleado con máscaras 1D: {time1D:0.9f}")
print(f"Factor 2D/1D: {time2D / time1D:0.2f}")
fig, axs = plt.subplots(1, 3, layout="constrained")
axs[0].imshow(imagen, cmap=plt.cm.gray)
axs[0].axis('off')
axs[1].plot(MASK_SIZES, times2D, 'r-', label='2D')
axs[1].plot(MASK_SIZES, times1D, 'b-', label='1D')
axs[2].axis('off')
plt.show()
