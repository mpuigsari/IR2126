"""""
Antes de ejecutar el programa preveo

Canal R
En este canal destacaran la parte roja de la bandera y la parte amarilla ya que está formada por rojo y verde. 
Además de la barra que mantiene la bandera ya que su gris se acerca al blanco que contiene todas las componentes

Canal G
En este canal destacara la parte amarilla ya que está formada por rojo y verde. 
Además de la barra que mantiene la bandera ya que su gris se acerca al blanco que contiene todas las componentes

Canal B
En este canal destacará el cielo y la parte azul marino de la bandera
Además de la barra que mantiene la bandera ya que su gris se acerca al blanco que contiene todas las componentes


Una vez ejecutado el programa he comprobado que se cumple todo lo dicho con la excepción de que el cielo parece tener
una gran componente verde además de azul.
"""""
import skimage as ski
import matplotlib.pyplot as plt

imagen_en_color = ski.io.imread("images/belgium_flag.jpg")

plano_rojo = imagen_en_color[:, :, 0]
plano_verde = imagen_en_color[:, :, 1]
plano_azul = imagen_en_color[:, :, 2]

fig, axs = plt.subplots(2, 3, layout="constrained")
axs[0, 1].imshow(imagen_en_color)
axs[1, 0].imshow(plano_rojo, cmap=plt.cm.gray)
axs[1, 1].imshow(plano_verde, cmap=plt.cm.gray)
axs[1, 2].imshow(plano_azul, cmap=plt.cm.gray)

for ax in axs.ravel():
    ax.set_axis_off()
plt.show()