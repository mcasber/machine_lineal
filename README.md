Descripción General

Este proyecto de machine learning tiene como objetivo entrenar dos algoritmos de aprendizaje supervisado, LinearRegression y RandomForestRegressor, para predecir valores y comparar sus métricas de desempeño. El sistema se actualiza diariamente mediante un proceso automatizado de web scraping y procesamiento de datos, permitiendo mantener el modelo actualizado y relevante.

Fecha de inicio: 2 de enero de 2024

Tecnologías utilizadas: Python | Scikit-learn | Matplotlib/Seaborn | BeautifulSoup | Undetected_chromedriver | Pandas
Bibliotecas adicionales: numpy, csv, subprocess

Estructura del proyecto

El proyecto está dividido en módulos principales que automatizan la recolección, procesamiento, y análisis de datos:

1. **Módulo **control

Este módulo se ejecuta diariamente como una tarea programada a las 10:00 a.m. Sus funciones principales son:
Ejecutar los scripts wscraping_dolar y wscraping_uva para realizar web scraping de datos relevantes (por ejemplo, tasas de cambio del dólar y valores de la unidad de valor adquisitivo - UVA).
Guardar los datos extraídos en archivos ubicados en la carpeta files.
Llamar al script consolidado mediante un subproceso para iniciar el procesamiento de datos recopilados.

Funcionamiento del Módulo control:
Importa las funciones dolar y uva de los scripts correspondientes.
Obtiene la fecha actual y los valores de las fuentes web.
Guarda los datos en archivos CSV (datos.csv para el dólar y uva.csv para la UVA).
Ejecuta el script consolidado.py como un subproceso para procesar y consolidar los datos.

Código principal:
from wscraping_dolar import dolar
from wscraping_uva import uva
import datetime
import csv
import os
import subprocess

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta = os.path.join(BASE_DIR, 'files', 'datos.csv')
ruta1 = os.path.join(BASE_DIR, 'files', 'uva.csv')

fecha = datetime.datetime.today().strftime('%d/%m/%y')

def guardar_dolar(fecha, precio, nombre_archivo='datos.csv'):
    with open(nombre_archivo, mode='a', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv, delimiter=',')
        escritor_csv.writerow([fecha, precio])

def guardar_uva(fecha, uva, nombre_archivo='uva.csv'):
    with open(nombre_archivo, mode='a', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv, delimiter=',')
        escritor_csv.writerow([fecha, uva])

if __name__ == '__main__':
    guardar_dolar(fecha, dolar, ruta)
    guardar_uva(fecha, uva, ruta1)
    print('----------> Iniciando script de consolidado de datos...')
    subprocess.Popen(['python', r'C:\Users\Mariano\Desktop\WebScraping\consolidado.py'])

2. **Script **wscraping_dolar

Este script realiza web scraping para obtener el precio de compra del dólar desde la página web DolarHoy:

Bibliotecas utilizadas:
BeautifulSoup para procesar y analizar el HTML.
Undetected_chromedriver para automatizar el navegador y evitar bloqueos por detección de bots.

Funcionamiento:
Abre el navegador con undetected_chromedriver y accede a la URL objetivo.
Extrae el código HTML de la página y utiliza BeautifulSoup para procesar el contenido.
Encuentra el valor de compra del dólar dentro de un elemento HTML con clase val.
Cierra el navegador y devuelve el dato extraído.

Código principal:
from bs4 import BeautifulSoup as bs
import datetime
import undetected_chromedriver as uc

browser = uc.Chrome()
url = 'https://dolarhoy.com/'
browser.get(url)
html = browser.page_source
soup = bs(html, 'html')
browser.close()
browser.quit()

compra = soup.find('div', {'class': 'val'}).text
dolar = int(compra[1:])
print(f'El precio de compra de USD es: ${dolar}')

3. **Script **wscraping_uva

Este script realiza web scraping para obtener el valor de la unidad de valor adquisitivo (UVA) desde la página web del Banco Central de la República Argentina:

Bibliotecas utilizadas:
BeautifulSoup para procesar y analizar el HTML.
Undetected_chromedriver para automatizar el navegador y evitar bloqueos por detección de bots.

Funcionamiento:
Abre el navegador con undetected_chromedriver y accede a la URL objetivo.
Extrae el código HTML de la página y utiliza BeautifulSoup para procesar el contenido.
Busca todos los elementos div con el atributo align="right".
Recopila los valores encontrados y selecciona el correspondiente al valor de la UVA.
Limpia y convierte el dato a formato numérico.
Cierra el navegador y devuelve el dato extraído.

Código principal:
from bs4 import BeautifulSoup as bs
import undetected_chromedriver as uc

browser = uc.Chrome()
url = 'https://www.bcra.gob.ar/'
browser.get(url)
html = browser.page_source
soup = bs(html, 'html')
browser.close()
browser.quit()

divs = soup.find_all('div', {'align': 'right'})
valores = [div.text.strip() for div in divs]
uva = valores[11]
uva = uva.replace('.', '')
uva = float(uva.replace(',', '.'))
print(f'El precio de la UVA es: ${uva}')

4. **Módulo **consolidado

Este módulo realiza las siguientes tareas:
Cargar los datos de los archivos generados por los scripts de web scraping.
Unificar y limpiar los datos para generar un conjunto de datos consolidado.
Preparar el conjunto de datos final necesario para entrenar los modelos.
Llamar al script modeloLinearRegression mediante un subproceso para realizar el entrenamiento y análisis de los modelos.

Código principal:
import os
import pandas as pd
from files.funciones import separa_decena
from files.funciones import separa_trimestre
import subprocess

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta = os.path.join(BASE_DIR, 'files', 'datos.csv')
ruta1 = os.path.join(BASE_DIR, 'files', 'uva.csv')

df = pd.read_csv(ruta,sep=",",header=0)
df1 = pd.read_csv(ruta1,sep=",",header=0) #genero el df para tomar la data uva

df=df.merge(df1,on='fecha')

df['dia']=df['fecha'].apply(lambda x : x.split('/')[0]).astype(int)
df['mes']=df['fecha'].apply(lambda x : x.split('/')[1]).astype(int) #esto lo podemos abrir por estacionalidad 'trimestre'
df['ano']=df['fecha'].apply(lambda x : x.split('/')[2]).astype(int) 

del df['fecha']

df['decena']=df['dia'].apply(separa_decena)
#mejor lo abro en 3 column para que no impacte el peso de si es 1, 2 o 3
df['decena_1'] = df['decena'].apply(lambda x: 1 if x == 1 else 0)
df['decena_2'] = df['decena'].apply(lambda x: 1 if x == 2 else 0)
df['decena_3'] = df['decena'].apply(lambda x: 1 if x == 3 else 0)
del df['decena'] #ahora elimino 'decena' para que no impacte

df['trimestre']=df['mes'].apply(separa_trimestre)
df['tri_1'] = df['trimestre'].apply(lambda x: 1 if x == 1 else 0)
df['tri_2'] = df['trimestre'].apply(lambda x: 1 if x == 2 else 0)
df['tri_3'] = df['trimestre'].apply(lambda x: 1 if x == 3 else 0)
df['tri_4'] = df['trimestre'].apply(lambda x: 1 if x == 4 else 0)
del df['trimestre'] #elimino 'trimestre' para que no impacte

df.to_csv('c:/Users/Mariano/Desktop/WebScraping/files/df_consolidado.csv', index=False) #guardo el df con las actualizaciones

print('----------> Iniciando entrenamiento del modelo')
subprocess.Popen(['python', r'C:\Users\Mariano\Desktop\WebScraping\modeloLinearRegression.py'])

5. **Módulo **modeloLinearRegression

Este módulo entrena y evalúa los modelos de aprendizaje supervisado:

Algoritmos utilizados:
LinearRegression
RandomForestRegressor

Métricas calculadas:
Error Cuadrático Medio (MSE)
Raíz del Error Cuadrático Medio (RMSE)
Coeficiente de determinación (R2)
Realiza validación cruzada con 5 particiones (cv=5) para evaluar la robustez de los modelos.

Genera un DataFrame que incluye:
Predicciones de cada modelo.
Diferencia entre las predicciones y los valores reales.
Indicador de qué modelo tuvo mejores aproximaciones en cada instancia.
Porcentaje de éxito de cada modelo en las predicciones.
Guarda los resultados en un archivo CSV para monitorear la evolución del proyecto.

Código principal:
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import csv
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta2 = os.path.join(BASE_DIR, 'files', 'df_consolidado.csv')
ruta3 = os.path.join(BASE_DIR, 'files', 'mse.csv')

df = pd.read_csv(ruta2,sep=",",header=0)

#COMIENZO A TRABAJAR CON EL MODELO:
X = df.drop(['precio','dia','mes','ano'],axis=1) 
y = df['precio'] 

#Divido los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('La cantidad de datos para entrenar es de: ',len(X_train),'\n y para testear es de: ',len(X_test))

#Creo y entreno el modelo con LinearRegression.
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test).round(5)

#Métricas
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print(f'El Error Cuadrático Medio (MSE) en el conjunto de prueba es: {mse:.4f}, su raiz (RMSE): {rmse:.4f} y r2: {r2:.4f} con LinearRegression')
val_cruz_lr=cross_val_score(model, X_train, y_train, cv=5).round(2)
print(f'El resultado de validacion cruzada de lr es: {val_cruz_lr}')

#Entreno el modelo de RANDOM FOREST para ir comparando
model_rf = RandomForestRegressor(n_estimators=800, random_state=42) #dejo n_estimators=800 que es el que mejor da de 100 a 1000
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test).round(5)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mse_rf ** 0.5
r2_rf = r2_score(y_test, y_pred_rf) #tener presente que no es una métrica para modelos no lineales
print(f'El Error Cuadrático Medio (MSE) en el conjunto de prueba es: {mse_rf:.4f}, su raiz (RMSE):{rmse_rf:.4f} y r2: {r2_rf:.4f} con RandomForestRegressor')
val_cruz_rf=cross_val_score(model_rf, X_train, y_train, cv=5).round(2)#.mean().round(5)
print(f'El resultado de validacion cruzada de rf es: {val_cruz_rf}')

#Armo el df con la prediccion en cada instancia:
df['precio_lr'] = model.predict(X)

df['diferencia']=df['precio']-df['precio_lr'] #genero la variable diferencia 
df['abs_dif_lr']=abs(df['diferencia']) #tomo el valor absoluto de la diferencia
df['desvio%_lr']=df['abs_dif_lr']/df['precio'] #para luego generar la diferencia %
df=df.drop(['diferencia'],axis=1) #elimino la variable diferencia que ya no la uso porque tengo el abs

#Repito con RANDOM FOREST para ir comparando
df['precio_rf'] = model_rf.predict(X)

df['diferencia']=df['precio']-df['precio_rf'] #genero la variable diferencia 
df['abs_dif_rf']=abs(df['diferencia']) #tomo el valor absoluto de la diferencia
df['desvio%_rf']=df['abs_dif_rf']/df['precio'] #para luego generar la diferencia %
df=df.drop(['diferencia'],axis=1) #elimino la variable diferencia que ya no la uso porque tengo el abs

df['aciertos_lr']=df.apply(lambda row: 1 if row['abs_dif_lr'] < row['abs_dif_rf'] else 0, axis=1)

porcentaje_aciertos_lr = (df['aciertos_lr'].sum() / len(df)) * 100
porcentaje_aciertos_rf = 100-porcentaje_aciertos_lr
print(f"El porcentaje de mejores aproximaciones al precio real de lr: {porcentaje_aciertos_lr:.2f}% tal que rf tiene {porcentaje_aciertos_rf:.2f}%")

#GUARDO el resultado final
df.to_csv('c:/Users/Mariano/Desktop/WebScraping/files/df_final.csv', index=False)

6. **Módulo **grafica

Este módulo genera visualizaciones basadas en los datos procesados y almacenados:

Bibliotecas utilizadas:
Matplotlib
Seaborn

Visualización: Utiliza seaborn para trazar tres líneas:
Precio Real (precio)
Predicción de Regresión Lineal (precio_lr)
Predicción de Random Forest (precio_rf)

Las líneas se trazan contra el índice del DataFrame, lo que permite comparar la evolución temporal de los precios reales con las predicciones de ambos modelos.

Código principal:
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta = os.path.join(BASE_DIR,'files','df_final.csv')

df = pd.read_csv(ruta,sep=",",header=0)

sns.set(style='darkgrid')
sns.lineplot(data=df, x=df.index, y='precio', label='Precio')
sns.lineplot(data=df, x=df.index, y='precio_lr', label='Precio_LRegression')
sns.lineplot(data=df, x=df.index, y='precio_rf', label='Precio_RForest')
#plt.xticks(rotation=75, fontsize=10)  # Rotar las etiquetas a 75 grados y cambiar el tamaño de la fuente
#plt.tight_layout()  # Ajusta el diseño para que no se corten las etiquetas
plt.show()