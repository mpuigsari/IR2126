import skimage as ski
import numpy as np

imagenOriginal = ski.io.imread("images/flecha_transparente.png")
flecha1 = imagenOriginal[:, :, 0:3]
flecha2 = np.rot90(flecha1)
flecha3 = np.rot90(flecha2)
flecha4 = np.rot90(flecha3)

secuencia = np.stack([flecha1, flecha4, flecha3, flecha2], axis=0)

ski.io.imsave("images/flecha.gif", secuencia, loop=0, fps=4)