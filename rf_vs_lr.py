print('Ejecutando script RandomForest versus LinearRegression.py')
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
import csv

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta2 = os.path.join(BASE_DIR, 'files', 'df_consolidado.csv')
ruta3 = os.path.join(BASE_DIR, 'files', 'mseII.csv')

df = pd.read_csv(ruta2, sep=",", header=0)

# Separar las características (X) y el target (y)
X = df.drop('precio', axis=1)
y = df['precio']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión lineal
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test).round(5)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f'El Error Cuadrático Medio (MSE) con Linear Regression es: {mse_lr:.2f}')

# Entrenar el modelo de Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test).round(5)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f'El Error Cuadrático Medio (MSE) con Random Forest es: {mse_rf:.2f}')

# Guardar los resultados en un CSV en una sola fila
fecha = datetime.datetime.today().strftime('%d/%m/%y')

def guardar_mse(fecha, mse_lr, mse_rf, nombre_archivo='mseII.csv'):
    with open(nombre_archivo, mode='a', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv, delimiter=',')
        escritor_csv.writerow([fecha, mse_lr, mse_rf])

guardar_mse(fecha, mse_lr, mse_rf, ruta3)

# Mostrar algunas predicciones
ejemplos = 5
print('Predicciones con Linear Regression:')
for i in range(ejemplos):
    print(f'Prediccion Linear Regression: {y_pred_lr[i]}, Precio real: {y_test.iloc[i]}')

print('Predicciones con Random Forest:')
for i in range(ejemplos):
    print(f'Prediccion Random Forest: {y_pred_rf[i]}, Precio real: {y_test.iloc[i]}')

#Analizar esto
rmse_lr = mse_lr ** 0.5
rmse_rf = mse_rf ** 0.5
print(f'El Error Cuadrático Medio (RMSE) con Linear Regression es: {rmse_lr:.2f}')
print(f'El Error Cuadrático Medio (RMSE) con Random Forest es: {rmse_rf:.2f}')
