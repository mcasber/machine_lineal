from bs4 import BeautifulSoup as bs #para extraer el html y analizarlo
import undetected_chromedriver as uc #para que la page web no detecte que soy un ordenador

browser = uc.Chrome() #creo el objeto pa navegar
url='https://www.bcra.gob.ar/' #le paso una web 
browser.get(url) #y le digo que vaya a esa web
#browser.implicitly_wait(10)

html=browser.page_source #con este metodo guardo en una variable el codigo html de la pagina
#print(html) #no es un formato muy prolijo para leer

#uso beautifulsoup para darle un formato mas espaciado y procesado para trabajar bien con el 
soup=bs(html,'html')
#print(f'esto es la data que bajo: \n {soup} \n y aca finaliza.')

browser.close()
browser.quit()

# Buscar todos los div con align="right"
divs = soup.find_all('div', {'align': 'right'})

# Recopilar todos los valores de los div encontrados
valores = [div.text.strip() for div in divs]
#for i in valores: print(i)
#print(valores)

# Si sabes que el valor que buscas siempre está en una posición específica, por ejemplo el primer valor:
uva = valores[11]
#print(uva)
uva = uva.replace('.', '')
#print(uva)
uva = float(uva.replace(',', '.'))
#print(uva)
#print(f'el precio del uva es:$ {uva}') #\n y hoy es: {(datetime.datetime.today().strftime("%H:%M:%S--%A %d/%m/%y"))}')
