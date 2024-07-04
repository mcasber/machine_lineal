'''Este modulo entrena el modelo. Utilizo algoritmo LinearRegression de la librería scikit-learn, para aprendizaje supervisado.
Un modelo de regresion lineal predice el valor de la variable dependiente "y" a partir de la variable independiente "x"'''

print('Ejecutando script modeloLinearRegression.py')
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
import csv

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta2 = os.path.join(BASE_DIR, 'files', 'df_consolidado.csv')
ruta3 = os.path.join(BASE_DIR, 'files', 'mse.csv')

df = pd.read_csv(ruta2,sep=",",header=0)

# Aca comienzo a trabajar con el modelo:
# Separar las características (X) y el target (y)
X = df.drop('precio',axis=1) #defino cuales son las variables que tengo para predecir
y = df['precio'] #defino que es lo que quiero predecir
# Divido los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X_train, X_test, y_train, y_test)

# Creo y entreno el modelo.
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test).round(5)

ejemplos=5
for i in range(ejemplos):
  print(f'Prediccion Linear Regression: {y_pred[i]}, Precio real: {y_test.iloc[i]}')

# Predecir el siguiente valor pasandole nuevos atributos
entrada=[[1049,4,7,2024,1,0,0]] #de incluir 'predecir' iria en indice=4
salida = model.predict(entrada)
# Mostrar el valor predicho
print(f'El valor predicho para la nueva entrada de data es: {salida[0]:.2f}')

#Calculo el Error Cuadrático Medio (MSE), y lo guardo en un csv.
mse = mean_squared_error(y_test, y_pred)
print(f'El Error Cuadrático Medio (MSE) en el conjunto de prueba es: {mse:.10f}')
promedio=df['precio'].mean()
#print(f'El promedio de column precio del df es: {promedio:.2f}')

fecha=datetime.datetime.today().strftime('%d/%m/%y')
def guardar_mse(fecha,mse,promedio,nombre_archivo='mse.csv'):
    # Abrir el archivo en modo de escritura
    with open(nombre_archivo, mode='a', newline='') as archivo_csv:
        # Crear un escritor CSV
        escritor_csv = csv.writer(archivo_csv, delimiter=',')
        # Escribir la fila en el archivo CSV
        escritor_csv.writerow([fecha,mse,promedio])

guardar_mse(fecha,mse,promedio,ruta3) #por ultimo guardo el mse para ir teniendo el historial
