import skimage as ski
import matplotlib.pyplot as plt
import math
from skimage.color.adapt_rgb import adapt_rgb,hsv_value

"""
Considera el decorador adapt_rgb con el parámetro hsv_value, que ya apareció en la práctica 2. 
Define una función random_noise_hsv que aplique la función random_noise sobre una imagen en color haciendo uso del decorador. 
Utiliza esta nueva función definida para añadir a la imagen ojo_azul.png las mismas cantidades de ruido sal y pimienta 
y gaussiano que en el ejercicio anterior. Muestra, de manera similar a como lo hiciste en el primer ejercicio, 
los resultados obtenidos en este caso. Comenta los resultados, resaltando de manera especial las diferencias 
entre los resultados correspondientes a los ejercicios 1 y 2."""


@adapt_rgb(hsv_value)
def random_noise_hsv(imagen,*args, **kwargs):
    return ski.util.random_noise(imagen, *args, **kwargs)


imagen = ski.io.imread("images/ojo_azul.png")
imagen = ski.util.img_as_float(imagen)

# Añdir ruido sal y pimiemta
sp_values = (0.01,  # 1%
             0.05,  # 5%
             0.10)  # 10%

sp_noise = []
mode = "s&p"
for i in range(len(sp_values)):
    img_noise = random_noise_hsv(imagen,amount=sp_values[i],mode=mode)
    sp_noise.append(img_noise)

gaussian_values = (0.001,  # sigma = 0.032
                   0.005,  # sigma = 0.071
                   0.010)  # sigma = 0.1
gaussian_noise = []
mode = "gaussian"
for i in range(len(gaussian_values)):
    img_noise = random_noise_hsv(imagen,var=gaussian_values[i],mode=mode)
    gaussian_noise.append(img_noise)


def mostrar_por_filtro(titulo, imagen, sp_values, gaussian_values, sp_images, gaussian_images):
    fig, axs = plt.subplots(2, len(sp_values) + 1, layout="constrained")
    fig.suptitle(titulo, size=24)
    axs[0, 0].imshow(imagen, cmap=plt.cm.gray)
    axs[0, 0].set_title("Original")

    for i in range(len(sp_values)):
        axs[0, i + 1].imshow(sp_images[i], cmap=plt.cm.gray)
        axs[0, i + 1].set_title(f"Ruido S&P {sp_values[i] * 100:0.0f}%")

    for i in range(len(gaussian_values)):
        axs[1, i + 1].imshow(gaussian_images[i], cmap=plt.cm.gray)
        axs[1, i + 1].set_title(f"Ruido Gaussiano $\\sigma$ = {math.sqrt(gaussian_values[i]):0.2f}")

    ax = axs.ravel()
    for a in ax:
        a.set_axis_off()
    plt.show()

mostrar_por_filtro("Imágenes con ruido", imagen, sp_values, gaussian_values, sp_noise, gaussian_noise)
