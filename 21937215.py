import concurrent.futures
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import multiprocessing
import math

def sec_mult(A, B): # f() que calcula la mult. en secuencial, como toda la vida se ha hecho
    C = [[0] * n_col_B for i in range(n_fil_A)] # Crear y poblar la matrix  C = A*B
    for i in range(n_fil_A): # Hago la multiplicacion de AxB = C, i para iterar sobre las filas de A
        for j in range(n_col_B): # j para iterar sobre las columnas de B
            for k in range(n_col_A): # k para iterar en C
                C[i][j] += A[i][k] * B[k][j] # Aqui se hace la multiplicación y guardo en C.
    return C # retorno la matriz C

def par_mult(A, B): # f() que prepara el reparto de trabajo para la mult. en paralelo
    n_cores = multiprocessing.cpu_count() # Obtengo los cores de mi pc
    size_col = math.ceil(n_col_B/n_cores) # Columnas  a procesar x c/cpre, ver Excel adjunto
    size_fil = math.ceil(n_fil_A/n_cores) # Filas a procesar x c/cpre, ver Excel adjunto
    MC = multiprocessing.RawArray('i', n_fil_A * n_col_B) # Array MC de memoria compartida donde se almacenaran los resultados, ver excel adjunto
    cores = [] # Array para guardar los cores y su trabajo
    for core in range(n_cores):# Asigno a cada core el trabajo que le toca, ver excel adjunto
        i_MC = min(core * size_fil, n_fil_A) # Calculo i para marcar inicio del trabajo del core en relacion a las filas
        f_MC = min((core + 1) * size_fil, n_fil_A) # Calculo f para marcar fin del trabajo del core, ver excel
        cores.append(multiprocessing.Process(target=par_core, args=(A, B, MC, i_MC, f_MC)))# Añado al Array los cores y su trabajo
    for core in cores:
        core.start()# Arranco y ejecuto el trabajo para c/ uno de los cores que tenga mi equipo, ver excel
    for core in cores:
        core.join()# Bloqueo cualquier llamada hasta que terminen su trabajo todos los cores
    C_2D = [[0] * n_col_B for i in range(n_fil_A)] # Convierto el array unidimensional MC en una matrix 2D (C_2D)
    for i in range(n_fil_A):# i para iterar sobre las filas de A
        for j in range(n_col_B):# j para iterar sobre las columnas de B
            C_2D[i][j] = MC[i*n_col_B + j] # Guardo el C_2D los datos del array MC
    return C_2D

def par_core(A, B, MC, i_MC, f_MC): # La tarea que hacen todos los cores
    for i in range(i_MC, f_MC): # Size representado en colores en el excel que itera sobre las filas en A
        for j in range(len(B[0])): # Size representado en colores en el excel que itera sobre las columnas en B
            for k in range(len(A[0])): # n_fil_B o lo que es l mismo el n_col_A
                MC[i*len(B[0]) + j] += A[i][k] * B[k][j]# Guarda resultado en MC[] de cada core

def execute_matrix ():
    print('Ejecutando primer ejercicio parte A...')
    if n_col_A != n_fil_B: raise Exception('Dimensiones no validas')  # Compruebo que se puedan multiplicar A y B
    print('De manera no paralela...')
    inicioS = time.time()  # empiezo a contar el tiempo
    print(sec_mult(A, B))  # Ejecuto multiplicacion secuencial
    finS = time.time()  # cuento el tiempo que tarda en ejecutarse el algoritmo de manera secuencial
    print('Time taken = {} seconds'.format(finS - inicioS)) # imprimo la diferencia
    print('------------------------------')
    print('De manera paralela...')
    inicioP = time.time()  # empiezo a contar el tiempo
    print(par_mult(A, B))  # Ejecuto multiplicacion paralela
    finP = time.time()  # cuento el tiempo que tarda en ejecutarse el algoritmo de manera paralela
    print('Time taken = {} seconds'.format(finP - inicioP)) # imprimo la diferencia

    print('\nMatriz  A y B se han multiplicado con exito en SECUENCIAL ha tardado ', finS - inicioS, ' y en PARALELO ', finP - inicioP) # imprimo la diferencia
    # creo un grafico de barras que me permita ver la diferencia entre los dos diagramas mas visualmente mediante la herramienta plot
    height = [finS-inicioS, finP-inicioP]
    bars = ('Secuencial', 'Paralelo')
    x_pos = np.arange(len(bars))
    plt.bar(x_pos, height)
    plt.xticks(x_pos, bars)
    plt.show()

def merge_sort_np(arr): # funcion para realizar el merge sort de un array cualquiera introducido de forma secuencial
    n = len(arr)  # averiguamos la longitud del array
    if n <= 1: # caso base, que solo haya un elemento
        return arr # retornamos el arr
    else: # sino es el caso realizamos las particiones
        half = int(n/2) # decretamos la mitad del array
        l, r = merge_sort_np(arr[:half]),merge_sort_np(arr[half:]) # incializamos la parte der y izq, correspondiente a cada parte del arr
        return merge_np(l,r) # llamamos a merge que se encargara de mezclar


def merge_np(*args): # funcion que se encarga de mezclar el lado izq y el dercho de forma secuencial
    # le pasamos el *args que nos eprmite psar cualquier numero determinado de variables aunque podriamos pasarle: (l,r)
    n = len(args) # averiguamos la longitud del array
    res = [] # incializamos un tipo abstracto de datos para almacenar los resultados
    p_l = p_r = 0 # incializamos dos punteros, uno para cada parte
    l, r = args[0] if n == 1 else args # numero de args que tiene el merge_np

    while p_l < len(l) and p_r < len(r): #nos metemos en el while siempre que los punteros esten entre el rango de cada parte
        # para evitar que el puntero de la izquierda, por ejemplo, se meta en la parte derecha
        if r[p_r] >= l[p_l]: # si el puntero actual del der es mayor al puntero actual del izq
            res.append(l[p_l]) # añadimos a res
            p_l += 1 # incrementamos el puntero zquierdo
        else: # si el puntero actual del der es menor al puntero actual del izq
            res.append(r[p_r]) # añadimos a res
            p_r += 1 # incrementamos el puntero derecho
    res.extend(l[p_l:]) #añadimos lo que queda
    res.extend(r[p_r:]) #añadimos lo que queda

    return res # retornamos nuestro resultado

def merge_sort_pools (arr): # funcion que hace el merge sort paralelo usando Pools del modulo multiprocessing
    cpu_cores = os.cpu_count() #contamos los cores de la maquina en la que se ejecuta el codigo
    workers = multiprocessing.Pool(processes=cpu_cores) # creamos nuestros workers usando Pools, un worker por cada cpu_core que tengamos
    t = int(math.ceil(float(len(arr))/cpu_cores)) # decretamos el tamaño que debería tener cada uno
    arr = [arr[i * t:(i + 1) * t] for i in range(cpu_cores)] # distribuimos el trabajo similar a los demas
    arr = workers.map(merge_sort_np,arr) # llamamos al mergesort secuencial para cada worker
    # cuando la lon del arr sea mayor que 1
    while len(arr) > 1:
        # nos creamos un aux y actualizamos el arr y sus posiciones par que esten ordenados
        aux = arr.pop() if len(arr) % 2 == 1 else None
        arr = [(arr[i], arr[i + 1]) for i in range(0, len(arr), 2)]
        # llamamos a la funcion map del modulo multiprocesing llamando a merge secuencial para mezclar los resultados
        arr = workers.map(merge_np, arr) + ([aux] if aux else [])
    return arr[0] # retornamos el arr

def execute_merge_sort():
    print('Ejecutando segundo ejercicio parte B...')
    print('De manera no paralela...')
    print('Array de ejemplo: ', arr_ex)
    startS = time.time()  # empiezo a contar el tiempo
    print('Array ordenado:', merge_sort_np(arr_ex)) # ejecuto e imprimo el merge secuencial
    finS = time.time() # cuento el tiempo que tarda en ejecutarse el algoritmo de manera secuencial
    print('Time taken = {} seconds'.format(finS - startS)) # imprimero la diferencia
    print('------------------------------')
    print('De manera paralela...')
    print('Array de ejemplo: ', arr_ex)
    startP = time.time()  # empiezo a contar el tiempo
    print('Array ordenado:', merge_sort_pools(arr_ex)) # ejecuto e imprimo el merge paralelo
    finP = time.time() # cuento el tiempo que tarda en ejecutarse el algoritmo de manera paralela
    print('Time taken = {} seconds'.format(finP - startP)) # imprimero la diferencia

    print('\nSe ha realizado el merge con exito, en SECUENCIAL ha tardado ', finS - startS,' y en PARALELO ', finP - startP)  # imprimo la diferencia
    # creo un grafico de barras que me permita ver la diferencia entre los dos diagramas mas visualmente mediante la herramienta plot
    height = [finS - startS, finP - startP]
    bars = ('Secuencial', 'Paralelo')
    x_pos = np.arange(len(bars))
    plt.bar(x_pos, height)
    plt.xticks(x_pos, bars)
    plt.show()

def fibo_np(num): # funcion que hace el fibo de un numero de manera secuencial usando recursividad
    if num <= 2: # algo asi como un caso base
        return 1; # retorno 1
    else: # sino vuelvo a ejecutar
        return (fibo_np(num-1)+fibo_np(num-2)) # llamo recursivamente y vuelvo a empezar

def fibo_multiprocessing(x): # funcion para imprimir el resultado del fibo_np
    # util para llamar en las implementaciones paralelas
    print('Fibo del {} resultado: {} '.format(x, fibo_np(x)))

def fibo_mult_process(): # funcion para hacer fibo en paralelo usando Process
    cpu_cores = os.cpu_count()# veo los cores con los que se cuenta
    processes = [] # tipo de datos para almacenar la info
    for i in range(cpu_cores): # for para recorrer por cada core
        core = multiprocessing.Process(target=fibo_multiprocessing, args=(i,)) # tarea que tiene que hacer cada core
        print(f'Core data: {core};') # imprimimos la info de cada core
        processes.append(core) # añadimos
        core.start()

    for p in processes:
        p.join()

def fibo_mult_pools(num): # implementacion de fibo paralelo usando Pools
    cpu_cores = multiprocessing.cpu_count() # veo los cores con los que se cuenta
    with concurrent.futures.ProcessPoolExecutor() as executor: # uso el modulo concurrent.futures.ProcessPoolExecutor() para crear un executor/capataz
        results = [executor.submit(fibo_multiprocessing, num) for _ in range(cpu_cores)] # hago el fibo del numero mediante submit, en base al numero de cores
        for f in concurrent.futures.as_completed(results):
            # imprimo los results que cada core ha creado
            return f.result()

def execute_fibo():
    print('Ejecutando segundo ejercicio parte C...')
    print('De manera no paralela...')
    number = int(input("Numero a realizar el fibo: "))
    startS = time.time() # empiezo a contar el tiempo
    print(fibo_np(number)) # ejecuto e imprimo el fibo secuencial
    finS = time.time() # cuento el tiempo que tarda en ejecutarse el algoritmo de manera secuencial
    print('Time taken = {} seconds'.format(finS - startS)) # imprimero la diferencia
    print('------------------------------')
    print('De manera paralela usando Process...')
    startP = time.time() # empiezo a contar el tiempo
    print(fibo_mult_process()) # ejecuto e imprimo el fibo paralelo usando Process
    print('-------------------')
    print('De manera paralela usando Pools...')
    print(fibo_mult_pools(number)) # ejecuto e imprimo el fibo paralelo usando Pools
    finP = time.time() # cuento el tiempo que tarda en ejecutarse el algoritmo de manera paralela
    print('Time taken = {} seconds'.format(finP - startP)) # imprimero la diferencia

    print('\nSe ha realizado el fibonacci con exito, en SECUENCIAL ha tardado ', finS - startS, ' y en PARALELO ',finP - startP)  # imprimo la diferencia
    # creo un grafico de barras que me permita ver la diferencia entre los dos diagramas mas visualmente mediante la herramienta plot
    height = [finS - startS, finP - startP]
    bars = ('Secuencial', 'Paralelo')
    x_pos = np.arange(len(bars))
    plt.bar(x_pos, height)
    plt.xticks(x_pos, bars)
    plt.show()

if __name__ == '__main__':
    # primero declaro e incializo las variables a usar
    cpu_cores = os.cpu_count()  # contamos los cores de la maquina en la que se ejecuta el codigo
    print(f'Antes de nada, su maquina tiene {cpu_cores} cores para usar.') # lo imprimimos
    correcto = True
    correo = input('Introduzca su correo: ')
    usuario = input('Introduza su usuario: ')
    A = [[random.randint(0, 219) for i in range(6)] for j in range(10)]  # Genero A[21937215][6]con num. aleatorios del 0 al 219, ver excel
    B = [[random.randint(0, 219) for i in range(200)] for j in range(6)]  # Genero B[6][21937215]con num. aleatorios del 0 al 219, ver excel
    n_fil_A = len(A)  # Obtengo num de filas de A
    n_col_A = len(A[0])  # Obtengo num de colunmas de A
    n_fil_B = len(B)  # Obtengo num de filas de B
    n_col_B = len(B[0])  # # Obtengo num de filas de B
    while correcto:
        # simulamos un correo y un numero de usuario cualquiera
        if (correo == 'user@gmail.com' and usuario == '21945383'):
            print(''' 
                ************UNIVERSIDAD EUROPEA************
                Escuela de Ingeniería Arquitectura y Diseño

                *******************MENU********************
                Miguel Ramos López                 21937215
                1.Ejercicio Parte A - Ejer Matriz 
                2.Ejercicio Parte B - Ejer MergeSort
                3.Ejercicio Parte C - Ejer Fibonacci
                4.Salir\n''')
            output = input('Seleccione: ')

            if (output == '1'):
                execute_matrix()
            if (output == '2'):
                elements = int(input("Introduce los elementos del array a ordenar: ")) # le pedimos al usuario que nos digite el tamaño del arr
                arr_ex = [random.randint(0, elements) for i in range(elements)] # lo creamos de forma aleatoria con los elementos que nos digito el user
                execute_merge_sort()
            if (output == '3'):
                execute_fibo()
            if (output == '4'): # si se escoge esta opcion salimos del programa y no se imprime mas el menu
                print('Saliste')
                correcto = False
        else: # opcion de que los datos introducidos a correo y usuario sean diferentes a los propuestos
            print('Datos incorrectos...')
            correcto = False