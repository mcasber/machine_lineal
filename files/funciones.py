#aca genero las funciones que voy a utilizar
def separa_decena(n):
    if n <= 10:
        return 1
    elif n <=20 and n > 10:
        return 2
    else:
        return 3

def separa_trimestre(n):
    if n <= 3:
        return 1
    elif n <=6 and n > 3:
        return 2
    elif n <=9 and n > 6:
        return 3
    else:
        return 4

print(separa_trimestre(10))