'''Este modulo se ejecuta como tarea programada a las 10 am todos los días.
Descarga datos de la web y los guarda en los respectivos archivos de la carpeta files, y luego ejecuta el script modelo'''

from wscraping_dolar import dolar
from wscraping_uva import uva
import datetime
import csv
import os
import subprocess

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta = os.path.join(BASE_DIR, 'files', 'datos.csv')
ruta1 = os.path.join(BASE_DIR, 'files', 'uva.csv')

fecha=datetime.datetime.today().strftime('%d/%m/%y')

def guardar_dolar(fecha, precio,nombre_archivo='datos.csv'):
    # Abrir el archivo en modo de escritura
    with open(nombre_archivo, mode='a', newline='') as archivo_csv:
        # Crear un escritor CSV
        escritor_csv = csv.writer(archivo_csv, delimiter=',')
        # Escribir la fila en el archivo CSV
        escritor_csv.writerow([fecha, precio])

def guardar_uva(fecha,uva,nombre_archivo='uva.csv'):
    # Abrir el archivo en modo de escritura
    with open(nombre_archivo, mode='a', newline='') as archivo_csv:
        # Crear un escritor CSV
        escritor_csv = csv.writer(archivo_csv, delimiter=',')
        # Escribir la fila en el archivo CSV
        escritor_csv.writerow([fecha, uva])

if __name__=='__main__':

    guardar_dolar(fecha,dolar,ruta)
    guardar_uva(fecha,uva,ruta1)
            
    # Llamar al otro script para levantar un proceso
    print('Iniciando otro proceso...')
    subprocess.Popen(['python', r'C:\Users\Mariano\Desktop\WebScraping\modelo.py'])
    print('Proceso finalizado')
