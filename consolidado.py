'''Este modulo levanta los datos de los respectivos archivos y los une.
Luego realiza algunos procesos para generar el set de datos final con el que se entrena el modelo.'''

print('Ejecutando script consolidado.py')
import os
import pandas as pd
from files.funciones import separa_decena
import subprocess

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta = os.path.join(BASE_DIR, 'files', 'datos.csv')
ruta1 = os.path.join(BASE_DIR, 'files', 'uva.csv')

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
df.to_csv('c:/Users/Mariano/Desktop/WebScraping/files/df_consolidado.csv', index=False)#guardo el df con las actualizaciones

# Llamar al otro script para levantar un proceso
print('----------> Iniciando entrenamiento del modelo')
subprocess.Popen(['python', r'C:\Users\Mariano\Desktop\WebScraping\modeloLinearRegression.py'])
