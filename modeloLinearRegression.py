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
print('La cantidad de datos para testear es de: ',len(y_pred))

ejemplos=5
for i in range(ejemplos):
  print(f'Prediccion Linear Regression: {y_pred[i]}, Precio real: {y_test.iloc[i]}')

#Predecir el siguiente valor pasandole nuevos atributos
#entrada=[[1049,4,7,2024,1,0,0]] #de incluir 'predecir' iria en indice=4
#salida = model.predict(entrada)
#print(f'El valor predicho para la nueva entrada de data es: {salida[0]:.2f}')# veo el valor predicho

#Calculo el Error Cuadrático Medio (MSE), y lo guardo en un csv.
mse = mean_squared_error(y_test, y_pred)
print(f'El Error Cuadrático Medio (MSE) en el conjunto de prueba es: {mse:.2f}')

#Armo el df con la prediccion en cada instancia y la diferencia
df['precio2'] = model.predict(X)
df['diferencia']=df['precio']-df['precio2']
print(df)

df.to_csv('c:/Users/Mariano/Desktop/WebScraping/files/df_final.csv', index=False)#guardo el resultado final

#Guardo el mse para ir teniendo el historial
fecha=datetime.datetime.today().strftime('%d/%m/%y')
def guardar_mse(fecha,mse,nombre_archivo='mse.csv'):
    # Abrir el archivo en modo de escritura
    with open(nombre_archivo, mode='a', newline='') as archivo_csv:
        # Crear un escritor CSV
        escritor_csv = csv.writer(archivo_csv, delimiter=',')
        # Escribir la fila en el archivo CSV
        escritor_csv.writerow([fecha,mse])

guardar_mse(fecha,mse,ruta3)
