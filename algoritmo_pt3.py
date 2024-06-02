from logging import exception
from funciones_apoyo import local_search_P1,calculate_fitness, generar_vecino
import random
import numpy as np

# Trayectorias Multiples
# BMB - Busqueda Multiarranque Basica
def BMB(datos_numpy_list, etiquetas_char_lists, num_iteraciones=20, funcion = local_search_P1):
    mejor_solucion = None
    mejor_fitness = 0

    for _ in range(num_iteraciones):
        solucion_inicial = generar_solucion_aleatoria(len(datos_numpy_list[0][0]))
        
        solucion_optima = funcion(datos_numpy_list, etiquetas_char_lists, solucion_inicial, max_evaluaciones=750)

        fitness = calculate_fitness(solucion_optima, datos_numpy_list, etiquetas_char_lists)

        if fitness > mejor_fitness:
            mejor_fitness = fitness
            mejor_solucion = solucion_optima

    return mejor_solucion

# ------------------------------------
# ILS - Iterated Local Search
def ILS(datos_numpy_list, etiquetas_char_lists, num_iteraciones=20, intensidad_mutacion=0.25, funcion = local_search_P1):
    # Generar solución inicial aleatoria
    solucion_inicial = generar_solucion_aleatoria(len(datos_numpy_list[0][0]))
    
    # Aplicar función a la solución inicial
    solucion = funcion( datos_numpy_list, etiquetas_char_lists, solucion_inicial, max_evaluaciones=750)
    mejor_solucion = solucion
    mejor_fitness = calculate_fitness(solucion, datos_numpy_list, etiquetas_char_lists)

    for _ in range(num_iteraciones):
        # Aplicar mutación a la mejor solución encontrada
        solucion_mutada = mutacion(mejor_solucion, intensidad_mutacion)
        
        # Aplicar búsqueda local a la solución mutada
        solucion_optima = funcion(datos_numpy_list, etiquetas_char_lists, solucion_mutada, max_evaluaciones=750)
        
        # Calcular fitness de la solución optimizada
        fitness = calculate_fitness(solucion_optima, datos_numpy_list, etiquetas_char_lists)
        
        # Actualizar la mejor solución encontrada si la nueva es mejor
        if fitness > mejor_fitness:
            mejor_fitness = fitness
            mejor_solucion = solucion_optima

    return mejor_solucion

# ------------------------------------
# Trayectorias Simples
# Algoritmo de Enfriamiento Simulado
def ES(datos_numpy_list, etiquetas_char_lists, solucion_actual = None,T0=None, Tf=1e-3, max_evaluaciones=15000,max_vecinos=None,max_exitos=None, instensidad_mutacion = 0.25, vecinos_por_enfriamiento = False):
    num_attributes = len(datos_numpy_list[0][0])
    # Generar solución inicial aleatoria
    if(max_vecinos is None):
        max_vecinos=10*num_attributes
    if(max_exitos is None):
        max_exitos=0.1*max_vecinos
    if(solucion_actual is None): 
        solucion_actual = generar_solucion_aleatoria(len(datos_numpy_list[0][0]))
    fitness_actual = calculate_fitness(solucion_actual, datos_numpy_list, etiquetas_char_lists)
    
    mejor_solucion = solucion_actual
    mejor_fitness = fitness_actual
    if(T0 is None):
        T0 = (fitness_actual * 0.1)/(-np.log(0.3))
    elif(T0 < Tf):
        exception("T0 debe ser mayor que Tf")

    T = T0
    num_evaluaciones = 0
    num_enfriamientos = 1
    
    while T > Tf and num_evaluaciones < max_evaluaciones:
        num_exitos = 0
        for _ in range(max_vecinos):
            if num_evaluaciones >= max_evaluaciones:
                break
            
            # Generar vecino
            if vecinos_por_enfriamiento:
                vecino = generar_vecino(num_attributes, solucion_actual, instensidad_mutacion)
            else:
                vecino = generar_vecino(num_attributes, solucion_actual, instensidad_mutacion,(Tf)/T)
            
            fitness_vecino = calculate_fitness(vecino, datos_numpy_list, etiquetas_char_lists)
            num_evaluaciones += 1
            
            
            delta =  fitness_actual - fitness_vecino
            if delta < 0 or random.random() <= probabilidad_aceptacion(delta, T):
                solucion_actual = vecino
                fitness_actual = fitness_vecino
                if fitness_vecino > mejor_fitness:
                    mejor_fitness = fitness_vecino
                    mejor_solucion = vecino
                num_exitos += 1
            
            if num_exitos >= max_exitos:
                break
        
        num_enfriamientos += 1
        # Enfriar la temperatura
        T = esquema_Cauchy(T,T0,Tf, num_enfriamientos)
    
    return mejor_solucion

def esquema_Cauchy(T,T0,Tf, M):
    B = (T0 - Tf) / (T0*Tf*M)
    T = T / (1 + T * B)
    return T


def generar_solucion_aleatoria(n):
    return np.random.uniform(0, 1, n)

# TODO t características
# Preguntar si se muta sumando un vector normalización (0, 0,3)
def mutacion(weights, s = 0.25, t = None):
    if(t is None):
        t = int(0.20 * len(weights))
    indices_mutacion = np.random.choice(len(weights), t, replace=False)
    for i in indices_mutacion:
        weights[i] += np.random.uniform(-s, s)
        weights[i] = np.clip(weights[i], 0, 1)
    return weights


def probabilidad_aceptacion(delta, T):
    if delta < 0:
        return 1.0  # Acepta soluciones mejores siempre
    else:
        return np.exp(-delta / T)
