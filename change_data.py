#DESDE ACA DISPARO LA TAREA PROGRAMADA PARA LAS 10:00 AM
from wscraping_dolar import precio_compra
import datetime
import csv
import os
import subprocess

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta = os.path.join(BASE_DIR, 'files', 'datos.csv')

fecha=datetime.datetime.today().strftime('%d/%m/%y')

def guardar_en_csv(fecha, precio,nombre_archivo='datos.csv'):
    # Nombre del archivo CSV
    ##nombre_archivo = 'datos.csv'

    # Abrir el archivo en modo de escritura
    with open(nombre_archivo, mode='a', newline='') as archivo_csv:
        # Crear un escritor CSV
        escritor_csv = csv.writer(archivo_csv, delimiter=',')

        # Escribir la fila en el archivo CSV
        escritor_csv.writerow([fecha, precio])

if __name__=='__main__':

    guardar_en_csv(fecha, precio_compra,ruta)
    
    # Llamar al otro script para levantar un proceso
    print('Iniciando otro proceso...')
    subprocess.Popen(['python', r'C:\Users\Mariano\Desktop\WebScraping\prediccion.py'])
    print('Proceso iniciado')
