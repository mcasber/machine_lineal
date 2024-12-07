import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score #para realizar validacion cruzada
from sklearn import preprocessing

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta2 = os.path.join(BASE_DIR, 'files', 'consolidado_pruebas.csv')
df = pd.read_csv(ruta2,sep=",",header=0)

# COMIENZO A TRABAJAR CON EL MODELO:
#df=df.drop(['dia','mes','ano'],axis=1) #elimino variables que no suman #ver que fallo aca

#Escalamos
scaler = preprocessing.MinMaxScaler()# Crear una instancia de MinMaxScaler
scaler.fit(df)# Ajustar el scaler a los datos
df_escalado = scaler.transform(df)# Transformar los datos usando el scaler ajustado
df = pd.DataFrame(df_escalado, columns=df.columns)# Convertir el ndarray escalado de nuevo a DataFrame

# Separar las características (X) y el target (y)
X = df.drop(['precio','dia','mes','ano'],axis=1) #defino cuales son las variables que tengo para predecir
y = df['precio'] #defino que es lo que quiero predecir
#------------------------------------------------------------------------------------------------------------
# Divido los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X_train, X_test, y_train, y_test)
print('La cantidad de datos para entrenar es de: ',len(X_train),'y para testear es de: ',len(X_test))
# Creo y entreno el modelo con LinearRegression.
model = LinearRegression()
model.fit(X_train, y_train)
# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test).round(5)
#Calculo el Error Cuadrático Medio (MSE) y el (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5 #la raíz del MSE (RMSE) para tener una métrica en la misma escala de los valores
# r2_score (coeficiente de determinacion), indica que es un buen modelo en general arriba de 75% , excelente arriba de los 80% 
## y casi perfecto por arriba de los 90%, 100% es que esta sobreajustado.
r2 = r2_score(y_test, y_pred)
print(f'El Error Cuadrático Medio (MSE) en el conjunto de prueba es: {mse:.4f}, su raiz (RMSE): {rmse:.4f} y r2: {r2:.4f} con LinearRegression')
val_cruz_lr=cross_val_score(model, X_train, y_train, cv=5).round(2)#.mean().round(5)
print(f'El resultado de validacion cruzada de lr es: {val_cruz_lr}')
    
ejemplos=3
for i in range(ejemplos):
  print(f'Prediccion Linear Regression: {y_pred[i]}, Precio real: {y_test.iloc[i]}')

    