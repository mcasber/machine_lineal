import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
ruta = os.path.join(BASE_DIR, 'uva.csv')

df1 = pd.read_csv(ruta,sep=",",header=0)

if __name__=='__main__':
        
    print(df1.info())
    print(df1.head(3))