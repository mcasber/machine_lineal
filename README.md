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

------------------------------------------------------------------------------------------------
Resultados y analisis con gpt 18/08/2024

1. **Error Cuadrático Medio (MSE) y RMSE**:
   - **Linear Regression**: MSE = 5660.20, RMSE = 75.23
   - **Random Forest**: MSE = 2247.22, RMSE = 47.40
   
   Aquí, `RandomForestRegressor` muestra un mejor rendimiento que `LinearRegression`, con un MSE y RMSE considerablemente menores. Esto sugiere que `RandomForestRegressor` está capturando mejor la relación entre las características y la variable objetivo.

2. **Validación Cruzada**:
   - **Linear Regression (LR)**: \( R^2 = 0.48275 \)
   - **Random Forest (RF)**: \( R^2 = 0.9626 \)
   
   Estos valores indican que `RandomForestRegressor` tiene un rendimiento mucho mejor en términos de la métrica \( R^2 \) (que mide la proporción de la varianza explicada por el modelo). Sin embargo, la diferencia es tan grande que es razonable preguntarse si el modelo de Random Forest podría estar sobreajustado.

### Reflexiones:

- **Gran diferencia en MSE/RMSE y \( R^2 \)**: La diferencia significativa entre los modelos sugiere que Random Forest está capturando patrones complejos en los datos que Linear Regression no puede capturar. Sin embargo, el hecho de que el \( R^2 \) sea tan alto para Random Forest, junto con el MSE relativamente bajo, podría indicar sobreajuste, especialmente si los datos de entrenamiento y prueba no son suficientemente diversos.

- **Validación Cruzada**: El alto valor de \( R^2 \) en la validación cruzada para Random Forest es consistente con el bajo MSE/RMSE, pero es importante recordar que, aunque estos resultados sean buenos, no garantizan que el modelo generalice bien en datos no vistos.

- **Prueba con más datos**: Si es posible, probar el modelo en un conjunto de datos completamente nuevo o aumentar el tamaño del conjunto de prueba podría ayudar a validar la robustez del modelo.

En resumen, los resultados son impresionantes para Random Forest, pero es recomendable realizar más pruebas, tal vez con datos adicionales o diferentes métricas, para asegurarse de que el modelo no esté simplemente sobreajustado a los datos actuales.

