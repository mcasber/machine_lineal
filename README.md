# Proyecto machine
Proyecto para tomar data de la web y aplicar un algoritmo de machine

Modulos:

control: Este modulo se ejecuta como tarea programada a las 10 am todos los días.
Descarga datos de la web y los guarda en los respectivos archivos de la carpeta files.
Luego ejecuta con subprocess el script consolidado.

consolidado:Este modulo levanta los datos de los respectivos archivos y los une.
Luego realiza algunos procesos para generar el set de datos final con el que se entrena el modelo.
Y finalmente ejecuta con subprocess el script modeloLinearRegression

modeloLinearRegression: Este modulo entrena el modelo. 
Utilizo el algoritmo LinearRegression de la librería scikit-learn, para aprendizaje supervisado.
Arma el df con la prediccion en cada instancia y la diferencia contra el valor real.
Calcula el Error Cuadrático Medio (MSE), y lo guarda en un csv para ir siguiendo la evolucion.

La carpeta files contiene todos los archivos de datos.

