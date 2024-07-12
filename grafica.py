import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta = os.path.join(BASE_DIR,'files','df_final.csv')

df = pd.read_csv(ruta,sep=",",header=0)
#print(df.head(2))

sns.set(style='darkgrid')
sns.lineplot(data=df, x=df.index, y='precio', label='Precio')
sns.lineplot(data=df, x=df.index, y='precio2', label='Precio2')
#plt.xticks(rotation=75, fontsize=10)  # Rotar las etiquetas a 75 grados y cambiar el tamaño de la fuente
#plt.tight_layout()  # Ajusta el diseño para que no se corten las etiquetas
plt.show()

'''
fig=px.line(df,x=df.index,y=['precio', 'precio2'],
            #labels={'fecha': 'Fecha', 'value':'Valores en $'},
            title='Relacion entre la variable precio y la prediccion/precio2',
            color_discrete_sequence=['blue', 'green'])
fig.show()'''

#print(df[60:80])