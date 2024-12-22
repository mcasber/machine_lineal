import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta = os.path.join(BASE_DIR,'files','mse.csv')

df = pd.read_csv(ruta,sep=",",header=0)
print(df.info())

sns.set(style='darkgrid')
sns.lineplot(data=df, x=df.index, y='rmse', label='LinearRegression_rmse')
sns.lineplot(data=df, x=df.index, y='rmse_rf', label='RandomForest_rmse')
#plt.xticks(rotation=75, fontsize=10)  # Rotar las etiquetas a 75 grados y cambiar el tamaño de la fuente
#plt.tight_layout()  # Ajusta el diseño para que no se corten las etiquetas
# Agregar títulos a los ejes
plt.title("Comparación de RMSE entre Linear Regression y Random Forest")  
plt.xlabel("Índice")  # Cambia "Índice" por el título que desees para el eje x
plt.ylabel("Root Mean Squared Error")  
plt.show()
