'''Este modulo levanta los datos de los respectivos archivos y los une.
Luego realiza algunos procesos para generar el set de datos final con el que se entrena el modelo.
Utilizo algoritmo LinearRegression de la librería scikit-learn, para aprendizaje supervisado.
Un modelo de regresion lineal predice el valor de la variable dependiente "y" a partir de la variable independiente "x"'''
print('Ejecutando script modelo.py')
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from files.funciones import separa_decena
import datetime
import csv

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta = os.path.join(BASE_DIR, 'files', 'datos.csv')
ruta1 = os.path.join(BASE_DIR, 'files', 'uva.csv')
ruta2 = os.path.join(BASE_DIR, 'files', 'mse.csv')

df = pd.read_csv(ruta,sep=",",header=0)
df1 = pd.read_csv(ruta1,sep=",",header=0) #genero el df para tomar la data uva

#ACA JOIN OTROS DF
df=df.merge(df1,on='fecha')
#print(df.info())

# !!!CON QUE LO QUIERO ENTRENAR!!!, creo y sumo atributos pa jugar
df['dia']=df['fecha'].apply(lambda x : x.split('/')[0]).astype(int)
df['mes']=df['fecha'].apply(lambda x : x.split('/')[1]).astype(int)
df['ano']=df['fecha'].apply(lambda x : x.split('/')[2]).astype(int)

del df['fecha'] #0   fecha     42 non-null     object  ----> ahora elimino esta columna porque no es un dato numerico

#ACA LE ESTOY SACANDO EL ATRIBUTO CLAVE  ----> o lo dejo para ver el comportamiento
#df['predecir']=df['precio']*1.015 #aplico % para jugar y ver como se comporta el mse
# 1.015 es aprox. para llegar al precio venta (el tomado como precio es compra)
##!! es un escalar y la variación es siempre igual --> testeado 29/06

#creo la column decena para separar en la etapa del mes
df['decena']=df['dia'].apply(separa_decena)

#mejor creo 3 column para que no impacte el peso de si es 1, 2 o 3 ? pensar esto
df['decena_1'] = df['decena'].apply(lambda x: 1 if x == 1 else 0)
df['decena_2'] = df['decena'].apply(lambda x: 1 if x == 2 else 0)
df['decena_3'] = df['decena'].apply(lambda x: 1 if x == 3 else 0)

del df['decena'] #ahora elimino 'decena' para que no impacte

print(df.info())

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
entrada=[[1050,29,6,2024,0,0,1]] #de incluir 'predecir' iria en indice=4
salida = model.predict(entrada)
# Mostrar el valor predicho
print(f'El valor predicho para la nueva entrada de data es: {salida[0]:.2f}')

#Calculo el Error Cuadrático Medio (MSE), y lo guardo en un csv.
mse = mean_squared_error(y_test, y_pred)
print(f'El Error Cuadrático Medio (MSE) en el conjunto de prueba es: {mse:.10f}')
promedio=df['precio'].mean()
print(f'El promedio de column precio del df es: {promedio:.2f}')

fecha=datetime.datetime.today().strftime('%d/%m/%y')
def guardar_mse(fecha,mse,promedio,nombre_archivo='mse.csv'):
    # Abrir el archivo en modo de escritura
    with open(nombre_archivo, mode='a', newline='') as archivo_csv:
        # Crear un escritor CSV
        escritor_csv = csv.writer(archivo_csv, delimiter=',')
        # Escribir la fila en el archivo CSV
        escritor_csv.writerow([fecha,mse,promedio])

guardar_mse(fecha,mse,promedio,ruta2) #por ultimo guardo el mse para ir teniendo el historial
