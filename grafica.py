import os
#from prediccion import df #ya no me sirve porque se elimino el campo 'fecha'
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta = os.path.join(BASE_DIR, 'files', 'datos.csv')

df = pd.read_csv(ruta,sep=",",header=0)

print(df.head(2))
sns.set(style='darkgrid')
sns.lineplot(data= df, x= 'fecha', y = 'precio')
plt.xticks(rotation=75, fontsize=10)  # Rotar las etiquetas a 75 grados y cambiar el tamaño de la fuente
plt.tight_layout()  # Ajusta el diseño para que no se corten las etiquetas
plt.show()