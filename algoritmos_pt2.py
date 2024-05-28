import random
import numpy as np
from funciones_apoyo import calculate_fitness, local_search

# TODO Para guardar fitness de cada individuo tengo que hacer algo.
# Hay que tener encuenta que la evaluación solo se tienen en cuenta al final 
# y a la hora de quedarse con el mejor. 
# Tengo guardar el mejor de cada generación.
# Hacer un vector de fitness y guardarlo en cada generación.

"""
Parameters:
- P: Array compuesto por N arrays de un número de carácterísticas. 

Returns:
- The final population after the algorithm has finished.

P -> es un vector que tiene en cada instancia dupla de un pesos y su fitness.
    Si no se tiene el fitness es null. Y al final se calcula el fitness de todos los null.

"""
### Genéticos ###
## 1
def algoritmo_genetico(P, datos_numpy_list, etiquetas_char_lists):
    t = 50
    evaluar(P, datos_numpy_list, etiquetas_char_lists) # Despues devolvera los fitness de todo. 
    # Y seleccionamos el mejor. Como P[i][0] es el peso y P[i][1] es el fitness.
    best_fitness, best_index = Best_fitness(P)
    # Mirar las evaluaciones de la función objetivo -> Mirar si uso atributo global
    # -> devuelve el número de evaluaciones.
    while t <= 15000:
        #print("Mejor P_prima: ", best_index, best_fitness,"\n" ,P[best_index][0])
        P_prima = seleccion_elitistas(P)
        ## Cruce / Recombinación
        for i in range(0, 34, 2):
            cromo_primero, cromo_segundo = cruce_BLX(P_prima[i][0].copy(), P_prima[i+1][0].copy())
            P_prima[i] = [cromo_primero, None]
            P_prima[i+1] = [cromo_segundo, None]
        ## Mutación
        seleccionados = random.choices(P_prima, k=4)
        for i in seleccionados:
            i[0] = mutar_generacional(i[0], 0.3)
            i[1] = None
        ## Evaluación
        t = t + evaluar(P_prima, datos_numpy_list, etiquetas_char_lists) # TODO Despues devolvera los fitnes de todo. Y seleccionamos el mejor y el peor.
        P,best_index,best_fitness  = reemplazar_generacional(P, P_prima,best_index,best_fitness)
    # Delvolver el mejor de la población.
    return P[best_index][0]
## 3
def algoritmo_genetico_estacionario(P, datos_numpy_list, etiquetas_char_lists):
    t = 50
    evaluar(P, datos_numpy_list, etiquetas_char_lists) # Despues devolvera los fitness de todo. 
    # Y seleccionamos el mejor. Como P[i][0] es el peso y P[i][1] es el fitness.
    best_fitness, best_index = Best_fitness(P)
    #print("Generación: ", t, best_fitness)
    # Mirar las evaluaciones de la función objetivo -> Mirar si uso atributo global
    # -> devuelve el número de evaluaciones.
    while t <= 15000:
        P_prima = seleccion_estacionaria(P)
        ## Cruce / Recombinación
        cromo_primero, cromo_segundo = cruce_BLX(P[0][0], P[1][0])
        P_prima[0] = [cromo_primero, None]
        P_prima[1] = [cromo_segundo, None]
        ## Mutación
        if random.random() < 0.1:
            P_prima[0] = [mutar_generacional(P_prima[0][0], 0.3), None]
        if random.random() < 0.1:
            P_prima[1] = [mutar_generacional(P_prima[1][0], 0.3), None]
        ## Evaluación
        t = t + evaluar(P_prima, datos_numpy_list, etiquetas_char_lists) # TODO Despues devolvera los fitnes de todo. Y seleccionamos el mejor y el peor.
        P = reemplazar_estacionario(P, P_prima)
    best_fitness, best_index = Best_fitness(P)
    # Delvolver el mejor de la población.
    return P[best_index][0]
## 2
def algoritmo_genetico_ARL(P, datos_numpy_list, etiquetas_char_lists):
    t = 50
    evaluar(P, datos_numpy_list, etiquetas_char_lists) # Despues devolvera los fitness de todo. 
    # Y seleccionamos el mejor. Como P[i][0] es el peso y P[i][1] es el fitness.
    best_fitness, best_index = Best_fitness(P)
    #print("Generación: ", t, best_fitness)
    # Mirar las evaluaciones de la función objetivo -> Mirar si uso atributo global
    # -> devuelve el número de evaluaciones.
    while t <= 15000:
        #print("Mejor P_prima: ", best_index, best_fitness,"\n" ,P[best_index][0])
        P_prima = seleccion_elitistas(P)
        ## Cruce / Recombinación
        for i in range(0, 34, 2):
            cromo_primero, cromo_segundo = cruce_aritmetico(P_prima[i][0], P_prima[i+1][0])
            P_prima[i] = [cromo_primero, None]
            P_prima[i+1] = [cromo_segundo, None]
        ## Mutación
        indices = random.sample(range(len(P_prima)), 4)
        for i in indices:
            P_prima[i] = [mutar_generacional(P_prima[i][0], 0.3), None]
        ## Evaluación
        t = t + evaluar(P_prima, datos_numpy_list, etiquetas_char_lists) # TODO Despues devolvera los fitnes de todo. Y seleccionamos el mejor y el peor.
        P,best_index,best_fitness  = reemplazar_generacional(P, P_prima,best_index,best_fitness)
        #print("Generación: ", t, best_fitness)
    # Delvolver el mejor de la población.
    return P[best_index][0]
## 4
def algoritmo_genetico_estacionario_AL(P, datos_numpy_list, etiquetas_char_lists):
    t = 50
    evaluar(P, datos_numpy_list, etiquetas_char_lists) # Despues devolvera los fitness de todo. 
    # Y seleccionamos el mejor. Como P[i][0] es el peso y P[i][1] es el fitness.
    best_fitness, best_index = Best_fitness(P)
    # print("Generación: ", t, best_fitness)
    # Mirar las evaluaciones de la función objetivo -> Mirar si uso atributo global
    # -> devuelve el número de evaluaciones.
    while t <= 15000:
        P_prima = seleccion_estacionaria(P)
        ## Cruce / Recombinación
        cromo_primero, cromo_segundo = cruce_aritmetico(P[0][0], P[1][0])
        P_prima[0] = [cromo_primero, None]
        P_prima[1] = [cromo_segundo, None]
        ## Mutación
        if random.random() < 0.1:
            P_prima[0] = [mutar_generacional(P_prima[0][0], 0.3), None]
        if random.random() < 0.1:
            P_prima[1] = [mutar_generacional(P_prima[1][0], 0.3), None]
        ## Evaluación
        t = t + evaluar(P_prima, datos_numpy_list, etiquetas_char_lists) # TODO Despues devolvera los fitnes de todo. Y seleccionamos el mejor y el peor.
        P = reemplazar_estacionario(P, P_prima)
        # print("Generación: ", t)
    best_fitness, best_index = Best_fitness(P)
    # Delvolver el mejor de la población.
    return P[best_index][0]

### Memeticos ###
def memetico_10(P, datos_numpy_list, etiquetas_char_lists):
    t = evaluar(P, datos_numpy_list, etiquetas_char_lists) # Despues devolvera los fitness de todo. 
    # Y seleccionamos el mejor. Como P[i][0] es el peso y P[i][1] es el fitness.
    best_fitness, best_index = Best_fitness(P)
    # Mirar las evaluaciones de la función objetivo -> Mirar si uso atributo global
    # -> devuelve el número de evaluaciones.
    while t <= 15000:
        g = 0
        while g < 10 and t <= 15000:
            # print("Mejor P_prima: ", best_index, best_fitness,"\n" ,P[best_index][0], "Funcion objetivo: ", t)
            P_prima = seleccion_elitistas(P)
            ## Cruce / Recombinación
            for i in range(0, 34, 2):
                cromo_primero, cromo_segundo = cruce_BLX(P_prima[i][0].copy(), P_prima[i+1][0].copy())
                P_prima[i] = [cromo_primero, None]
                P_prima[i+1] = [cromo_segundo, None]
            ## Mutación
            seleccionados = random.choices(P_prima, k=4)
            for i in seleccionados:
                i[0] = mutar_generacional(i[0], 0.3)
                i[1] = None
            ## Evaluación
            t = t + evaluar(P_prima, datos_numpy_list, etiquetas_char_lists) # TODO Despues devolvera los fitnes de todo. Y seleccionamos el mejor y el peor.
            P,best_index,best_fitness  = reemplazar_generacional(P, P_prima,best_index,best_fitness)
            g=g+1
        i = 0
        while i < 50 and t <= 15000:
            t = local_search(P[i],datos_numpy_list, etiquetas_char_lists,t)
            i = i + 1
    best_fitness, best_index = Best_fitness(P)
    # Delvolver el mejor de la población.
    return P[best_index][0]

def memetico_01(P, datos_numpy_list, etiquetas_char_lists):
    t = evaluar(P, datos_numpy_list, etiquetas_char_lists) # Despues devolvera los fitness de todo. 
        # Y seleccionamos el mejor. Como P[i][0] es el peso y P[i][1] es el fitness.
    best_fitness, best_index = Best_fitness(P)
        # Mirar las evaluaciones de la función objetivo -> Mirar si uso atributo global
        # -> devuelve el número de evaluaciones.
    while t <= 15000:
        g = 0
        while g < 10 and t <= 15000:
            # print("Mejor P_prima: ", best_index, best_fitness,"\n" ,P[best_index][0], "Funcion objetivo: ", t)
            P_prima = seleccion_elitistas(P)
            ## Cruce / Recombinación
            for i in range(0, 34, 2):
                cromo_primero, cromo_segundo = cruce_BLX(P_prima[i][0].copy(), P_prima[i+1][0].copy())
                P_prima[i] = [cromo_primero, None]
                P_prima[i+1] = [cromo_segundo, None]
            ## Mutación
            seleccionados = random.choices(P_prima, k=4)
            for i in seleccionados:
                i[0] = mutar_generacional(i[0], 0.3)
                i[1] = None
            ## Evaluación
            t = t + evaluar(P_prima, datos_numpy_list, etiquetas_char_lists) # TODO Despues devolvera los fitnes de todo. Y seleccionamos el mejor y el peor.
            P,best_index,best_fitness  = reemplazar_generacional(P, P_prima,best_index,best_fitness)
            g=g+1
        i = 0
        if t <= 15000:
            seleccionados = random.choices(P, k=5)
            for j in seleccionados:
                t = local_search(j,datos_numpy_list, etiquetas_char_lists,t)
            i = i + 1
    best_fitness, best_index = Best_fitness(P)
    # Delvolver el mejor de la población.
    return P[best_index][0]

def memetico_mej(P, datos_numpy_list, etiquetas_char_lists):
    t = evaluar(P, datos_numpy_list, etiquetas_char_lists)
    best_fitness, best_index = Best_fitness(P)
    while t <= 15000:
        g = 0
        while g < 10 and t <= 15000:
            # print("Mejor P_prima: ", best_index, best_fitness,"\n" ,P[best_index][0], "Funcion objetivo: ", t)
            P_prima = seleccion_elitistas(P)
            ## Cruce / Recombinación
            for i in range(0, 34, 2):
                cromo_primero, cromo_segundo = cruce_BLX(P_prima[i][0].copy(), P_prima[i+1][0].copy())
                P_prima[i] = [cromo_primero, None]
                P_prima[i+1] = [cromo_segundo, None]
            ## Mutación
            seleccionados = random.choices(P_prima, k=4)
            for i in seleccionados:
                i[0] = mutar_generacional(i[0], 0.3)
                i[1] = None
            ## Evaluación
            t = t + evaluar(P_prima, datos_numpy_list, etiquetas_char_lists)
            P,best_index,best_fitness  = reemplazar_generacional(P, P_prima,best_index,best_fitness)
            g=g+1
        # Después de las 10 generaciones, seleccionamos los 5 mejores individuos
        P.sort(key=lambda x: x[1], reverse=True)  # Asegúrate de que estás ordenando por fitness
        seleccionados = P[:5]
        for i in seleccionados:
            t = local_search(i,datos_numpy_list, etiquetas_char_lists,t)
    best_fitness, best_index = Best_fitness(P)
    return P[best_index][0]


### Funciones auxiliares ###

def Best_fitness(P):
    best_fitness = P[0][1]
    best_index = 0
    for i in range(1, len(P)):
        if P[i][1] > best_fitness:
            best_fitness = P[i][1]
            best_index = i
    return best_fitness, best_index

def Worst_fitness(P):
    worst_fitness = P[0][1]
    worst_index = 0
    for i in range(1, len(P)):
        if P[i][1] < worst_fitness:
            worst_fitness = P[i][1]
            worst_index = i
    return worst_fitness, worst_index

# Tiene que devolver un vector con todas las puntuaciones en el mismo orden que P
def evaluar(P, datos_numpy_list, etiquetas_char_list):
    count = 0
    for i in range(len(P)):
        if P[i][1] == None:
            P[i][1]=calculate_fitness(P[i][0], datos_numpy_list, etiquetas_char_list)
            count += 1
    return count

# SELECCIÓN ESTACIONARIO : se aplicará dos veces el torneo 
# para elegir los dos padres que serán posteriormente recombinados (cruzados).
def seleccion_estacionaria(P):
    P_prima = []
    
    indices = random.sample(range(len(P)), 3)
    mejores = max(indices, key=lambda i: P[i][1])
    P_prima.append(P[mejores].copy())
    
    indices = random.sample(range(len(P)), 3)
    mejores = max(indices, key=lambda i: P[i][1])
    P_prima.append(P[mejores].copy())
    return P_prima

# Un for del tamaño del p. Selecciona 3 elementos aleatorios y elige el mejor de los 3.
# Hasta devolver un array de tamaño P con los mejores.
def seleccion_elitistas(P):
    P_prima = []
    for _ in range(len(P)):
        seleccionados = random.choices(P, k=3)
        mejor = max(seleccionados, key=lambda x: x[1])
        P_prima.append(mejor.copy())
    return P_prima

# Se van seleccionando el pares concretamente en 17 pares
# Cada 2 cromosomas se recombinan. es decir el 0 con el 1, el 2 con el 3, etc. hasta el 34 que se para.
# Seguir los dos algoritmos de recombinación prpopuestos en el enunciado.
def cruce_BLX(cromo_a, cromo_b):
    alpha = 0.3
    cromo_min = np.minimum(cromo_a, cromo_b)
    cromo_max = np.maximum(cromo_a, cromo_b)
    L = cromo_max - cromo_min
    # todo el clip donde va en maximo minimo o en el cromo_a y cromo_b
    minimo = cromo_min - (L * alpha)
    maximo = cromo_max + (L * alpha)
    cromo_primero = np.clip(np.random.uniform(minimo, maximo),0,1)
    cromo_segundo = np.clip(np.random.uniform(minimo, maximo),0,1)
    return cromo_primero, cromo_segundo

def cruce_aritmetico(cromo_a, cromo_b):
    alpha = np.random.uniform(0.0, 1.0)
    cromo_primero = cromo_a * alpha + cromo_b * (1 - alpha)
    cromo_segundo = cromo_a * (1 - alpha) + cromo_b * alpha
    return cromo_primero, cromo_segundo

# MUTACIÓN ESTACIONARIO : tiramos dados no queda otra...

# Preguntar si se muta sumando un vector normalización (0, 0,3)
def mutar_generacional(weights, s):
    P_mutado = np.copy(weights)  # Establece la semilla del generador de números aleatorios    # Generate a random neighbor by adding a vector Z generated from a normal distribution
    Z = np.random.normal(0, s, len(weights))
    P_mutado += Z
    # Truncate the components to ensure they stay within the range [0, 1]
    P_mutado = np.clip(P_mutado, 0, 1)
    
    return P_mutado


# REMPLAZAR ESTACIONARIO : Muy diferente. Los hijos compiten para entrar en la población.
# todo Se puede guardar el peor de una generación a otra. Se puede hacer hasta una estructura
def reemplazar_estacionario(P, P_prima):
    worst_fitness = P[0][1]
    worst_fitness2 = P[1][1]
    worst_index = 0
    worst_index2 = 1
    for i in range(1, len(P)):
        if P[i][1] < worst_fitness:
            worst_index2 = worst_index
            worst_fitness2 = worst_fitness
            worst_fitness = P[i][1]
            worst_index = i
    
    if P_prima[0][1] > P_prima[1][1]:
        x1 = P_prima[0]
        x2 = P_prima[1]
    else:
        x1 = P_prima[1]
        x2 = P_prima[0]
    # Los dos están ordenados de mejor a peor. Por lo que si x1 es mejor que el segundo peor de P
    # x2 puede ser mejor que el segundo peor de P. Entrando los dos cromosomas. Si no solo entra x1.
    # Si no, si x1 es mejor que el peor de P, x1 entra en la población. En caso que x1 sea peor que el peor no entra ninguno.
    if x1[1] > worst_fitness2:
        if x2[1] > worst_fitness2:
            P[worst_index] = x2
            P[worst_index2] = x1
        else:
            P[worst_index] = x1
    elif x1[1] > worst_fitness:
        P[worst_index] = x1
    return P

# Seleccionar el mejor y el peor de la población y reemplazarlo 
# todo exepcto el mejor de la dos generaciones. Si el mejor es
# de la generación antingua sustituir por el peor en caso 
# contrario P = P_prima.
def reemplazar_generacional(P, P_prima, best_index,best_fitness):
    # Select the best individual from P_prima
    BestNewFitness, BestNewIndex = Best_fitness(P_prima)
    # Select the worst individual from P
    worst_fitness, worst_index= Worst_fitness(P_prima)
    # todo si es <= o <
    if(BestNewFitness < best_fitness):
        BestNewIndex = worst_index
        BestNewFitness = best_fitness
        P_prima[worst_index] = P[best_index] # SIEMPRE SE MANTIENE EL MEJOR DE LA GENERACIÓN ANTERIOR
    return P_prima, BestNewIndex, BestNewFitness