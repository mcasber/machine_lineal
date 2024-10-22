'''En este modulo generamos el archivo con los datos que queremos testear y nunca fueron mostrados al modelo.
Para ver que predicicones realiza y comparar con los datos reales'''

import os
import pandas as pd
import csv

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta = os.path.join(BASE_DIR,'df_consolidado.csv')

df = pd.read_csv(ruta,sep=",",header=0)
print(df.head(3))


#me quiero quedar con los últimos 10 precios y lo guardo en el csv 
precios_reales = df.iloc[-10:,0]
print(precios_reales)
precios_reales.to_csv('c:/Users/Mariano/Desktop/WebScraping/files/df_test.csv', index=False)

#ahora genero el arreglo que le voy a pasar para que realice la prediccion 
data_predict = df.iloc[-10:][['uva', 'decena_1', 'decena_2', 'decena_3', 'tri_1', 'tri_2', 'tri_3', 'tri_4']]
print(data_predict)
data_predict.to_csv('c:/Users/Mariano/Desktop/WebScraping/files/df_predict.csv', index=False)
