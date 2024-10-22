'''Este modulo entrena el modelo. Utilizo algoritmo LinearRegression de la librería scikit-learn, para aprendizaje supervisado.
Un modelo de regresion lineal predice el valor de la variable dependiente "y" a partir de la variable independiente "x"'''

print('Ejecutando script modeloLinearRegression.py')
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import csv
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score #para realizar validacion cruzada

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta2 = os.path.join(BASE_DIR, 'files', 'df_consolidado.csv')
ruta3 = os.path.join(BASE_DIR, 'files', 'mse.csv')

df = pd.read_csv(ruta2,sep=",",header=0)

#------------------------------------------------------------------------------------------------------------
#ESTO ES PARA VER LA CORRELACION ENTRE LAS VARIABLES:
#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.heatmap(df.drop(['dia','mes','ano'],axis=1).corr(), annot=True)
#plt.show()

#------------------------------------------------------------------------------------------------------------
# COMIENZO A TRABAJAR CON EL MODELO:
# Separar las características (X) y el target (y)
X = df.drop(['precio','dia','mes','ano'],axis=1) #defino cuales son las variables que tengo para predecir
y = df['precio'] #defino que es lo que quiero predecir

#------------------------------------------------------------------------------------------------------------
# Divido los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X_train, X_test, y_train, y_test)

#------------------------------------------------------------------------------------------------------------
# Creo y entreno el modelo con LinearRegression.
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test).round(5)
print('La cantidad de datos para testear es de: ',len(y_pred))

#Calculo el Error Cuadrático Medio (MSE) y el (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5 #la raíz del MSE (RMSE) para tener una métrica en la misma escala de los valores
# r2_score (coeficiente de determinacion), indica que es un buen modelo en general arriba de 75% , excelente arriba de los 80% 
## y casi perfecto por arriba de los 90%, 100% es que esta sobreajustado.
r2 = r2_score(y_test, y_pred)
print(f'El Error Cuadrático Medio (MSE) en el conjunto de prueba es: {mse:.4f}, su raiz (RMSE): {rmse:.4f} y r2: {r2:.4f}')

#------------------------------------------------------------------------------------------------------------
# Entreno el modelo de RANDOM FOREST para ir comparando
model_rf = RandomForestRegressor(n_estimators=800, random_state=42) #dejo n_estimators=800 que es el que mejor da de 100 a 1000
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test).round(5)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mse_rf ** 0.5 #la raíz del MSE (RMSE) para tener una métrica en la misma escala de los valores
r2_rf = r2_score(y_test, y_pred_rf)
print(f'El Error Cuadrático Medio (MSE) en el conjunto de prueba es: {mse_rf:.4f}, su raiz (RMSE): {rmse_rf:.4f}  y r2: {r2_rf:.4f} con RANDOM FOREST')
#------------------------------------------------------------------------------------------------------------

# COMIENZO A GENERAR DATA

#ejemplos=5
#for i in range(ejemplos):
#  print(f'Prediccion Linear Regression: {y_pred[i]}, Precio real: {y_test.iloc[i]}')

#Predecir el siguiente valor pasandole nuevos atributos
entrada=[[1143.59,1,0,0,0,0,1,0]]
salida = model_rf.predict(entrada)
print(f'El valor predicho para la nueva entrada de data es: {salida[0]:.2f}') # veo el valor predicho

#------------------------------------------------------------------------------------------------------------
#Armo el df con la prediccion en cada instancia:
df['precio_lr'] = model.predict(X)

df['diferencia']=df['precio']-df['precio_lr'] #genero la variable diferencia 
df['abs_dif_lr']=abs(df['diferencia']) #tomo el valor absoluto de la diferencia
df['desvio%_lr']=df['abs_dif_lr']/df['precio'] #para luego generar la diferencia %
df=df.drop(['diferencia'],axis=1) #elimino la variable diferencia que ya no la uso porque tengo el abs

#------------------------------------------------------------------------------------------------------------
#Repito con RANDOM FOREST para ir comparando
df['precio_rf'] = model_rf.predict(X)

df['diferencia']=df['precio']-df['precio_rf'] #genero la variable diferencia 
df['abs_dif_rf']=abs(df['diferencia']) #tomo el valor absoluto de la diferencia
df['desvio%_rf']=df['abs_dif_rf']/df['precio'] #para luego generar la diferencia %
df=df.drop(['diferencia'],axis=1) #elimino la variable diferencia que ya no la uso porque tengo el abs
#------------------------------------------------------------------------------------------------------------
print(df)
#------------------------------------------------------------------------------------------------------------
#VALIDACION CRUZADA
val_cruz_lr=cross_val_score(model, X_train, y_train, cv=5).mean().round(5)
print(f'El resultado de validacion cruzada de lr es: {val_cruz_lr}')
val_cruz_rf=cross_val_score(model_rf, X_train, y_train, cv=5).mean().round(5)
print(f'El resultado de validacion cruzada de rf es: {val_cruz_rf}')
#------------------------------------------------------------------------------------------------------------

# GUARDO el resultado final
df.to_csv('c:/Users/Mariano/Desktop/WebScraping/files/df_final.csv', index=False)

# GUARDO el mse para ir teniendo el historial
fecha=datetime.datetime.today().strftime('%d/%m/%y')
def guardar_mse(fecha,mse,rmse,mse_rf,rmse_rf,nombre_archivo='mse.csv'):
    # Abrir el archivo en modo de escritura
    with open(nombre_archivo, mode='a', newline='') as archivo_csv:
        # Crear un escritor CSV
        escritor_csv = csv.writer(archivo_csv, delimiter=',')
        # Escribir la fila en el archivo CSV
        escritor_csv.writerow([fecha,mse,rmse,mse_rf,rmse_rf])

guardar_mse(fecha,mse,rmse,mse_rf,rmse_rf,ruta3)
