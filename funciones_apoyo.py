from k_NN import KNNClassifier
import numpy as np

def calculate_fitness(weights, attributes, labels, alpha = 0.8):
    new_attributes = attributes.copy()
    weight = weights.copy()
    # Cuenta todos los valores de weight que son menores que 0.1
    longitud_instancias_menores = 0
    weight[weight < 0.1] = 0
    longitud_instancias_menores = np.count_nonzero(weight == 0)
    
    # Calcular la tasa de reducción
    tasa_reduccion = (longitud_instancias_menores / weight.size)* 100
    
    # Calcular la tasa de clasificación
    tasa_clas = accuracy(new_attributes, labels, weight)
    
    return ((alpha * tasa_clas) + ((1 - alpha) * tasa_reduccion))

# Devuelve tasa_clas.

def accuracy(datos_numpy_list, etiquetas_char_list, weight):
    porcentaje_coincidencia_media = 0
    clf = KNNClassifier(k=1)

    for i, (datos_numpy, etiquetas_char) in enumerate(zip(datos_numpy_list, etiquetas_char_list)):
        # Convertir las etiquetas a números

        # Concatenación de datos y etiquetas de otros archivos
        datos_entrenamiento = np.concatenate(
            [datos_numpy_list[(i+1) % 5] for _ in range(4)]
        )
        etiquetas_entrenamiento = np.concatenate(
            [etiquetas_char_list[(i+1) % 5] for _ in range(4)]
        )
        
        clf.fit(datos_entrenamiento, etiquetas_entrenamiento)
    
        predictions = clf.predict(datos_numpy, weight)
    
        porcentaje_coincidencia = (np.mean(predictions == etiquetas_char))*100
        porcentaje_coincidencia_media += porcentaje_coincidencia
    return porcentaje_coincidencia_media/5


### Busqueda Local ###
def local_search_P1(datos_numpy_list, etiquetas_char_lis, weights):
    num_attributes = len(datos_numpy_list[0][0])
    best_fitness = calculate_fitness(weights,datos_numpy_list, etiquetas_char_lis)

    # todo Se detendrá la ejecución cuando no se encuentre mejora tras generar un máximo de 
    # todo 20·n vecinos consecutivos (n es el número de características) o cuando se hayan realizado 
    # todo 15000 evaluaciones de la función objetivo, es decir, en cuanto se cumpla alguna de las 
    # todo dos condiciones.
    i = 0
    while i <= 750:
        # Perform local search by generating a neighbor instance
        neighbor = generar_vecino(num_attributes, weights, 0.3)
        
        neighbor_fitness = calculate_fitness( neighbor, datos_numpy_list, etiquetas_char_lis)

        i += 1
        if neighbor_fitness > best_fitness:
            weights = np.copy(neighbor)
            best_fitness = neighbor_fitness
    return weights

def local_search(w,datos_numpy_list, etiquetas_char_lis, i):
    num_attributes = len(datos_numpy_list[0][0])
    weights = w[0].copy()
    best_fitness = w[1].copy()

    # todo Se detendrá la ejecución cuando no se encuentre mejora tras generar un máximo de 
    # todo 20·n vecinos consecutivos (n es el número de características) o cuando se hayan realizado 
    # todo 15000 evaluaciones de la función objetivo, es decir, en cuanto se cumpla alguna de las 
    # todo dos condiciones.
    max_neighbors = 2 * num_attributes
    j = 0
    while i <= 15000 and j <= max_neighbors:
        # Perform local search by generating a neighbor instance
        neighbor = generar_vecino(num_attributes, weights, 0.3)
        
        neighbor_fitness = calculate_fitness( neighbor, datos_numpy_list, etiquetas_char_lis)

        i += 1
        j += 1
        if neighbor_fitness > best_fitness:
            weights = np.copy(neighbor)
            best_fitness = neighbor_fitness
            j = 0
    w[0] = weights
    w[1] = best_fitness
    return i
# Le estoy pasando la variación estandar en vez de la varianza.
def generar_vecino(num_attributes, weights, s):
    neighbor = np.copy(weights)  # Establece la semilla del generador de números aleatorios    # Generate a random neighbor by adding a vector Z generated from a normal distribution
    Z = np.random.normal(0, s, num_attributes)
    neighbor += Z
    # Truncate the components to ensure they stay within the range [0, 1]
    neighbor = np.clip(neighbor, 0, 1)
    
    return neighbor

def calculate_weights(neighbor, attributes, labels):
    new_attributes = [np.copy(datos_numpy) for datos_numpy in attributes]
    
    for i, datos_numpy in enumerate(new_attributes):
        for j, datos in enumerate(datos_numpy):
            for z in range(len(datos)):
                new_attributes[i][j][z] = new_attributes[i][j][z] * neighbor[z]  
    return accuracy(new_attributes, labels)