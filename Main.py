# Clase de datos para el problema
# Primero normalizar datos.
import numpy as np
from scipy.io import arff
import time
from algoritmos_pt2 import algoritmo_genetico, algoritmo_genetico_estacionario, algoritmo_genetico_ARL, memetico_10, memetico_01, memetico_mej
from algoritmo_pt3 import BMB, ILS, ES
from funciones_apoyo import local_search_P1
import sys
from sklearn.preprocessing import LabelEncoder
from k_NN import KNNClassifier
import random 
import pandas as pd
import datetime
import os

seed = 20
# Define listas para almacenar los datos y etiquetas de cada archivo
datos_numpy_list = []
etiquetas_char_list = []

valid_num1 = False
valid_num2 = False

if __name__ == "__main__":
    if len(sys.argv) != 4 and len(sys.argv) != 3:
        print("Invalid number of arguments. Please provide num1 and num2 as command line arguments.")
        sys.exit(1)
    
    num1 = int(sys.argv[1])
    num2 = int(sys.argv[2])
    if len(sys.argv) == 4:
        name = str(sys.argv[3])
    else:
        now = datetime.datetime.now()
        name = "results_" + now.strftime("%Y-%m-%d")
    
if num1 == 1:
    fnames = ['Instancias_APC/breast-cancer_1.arff', 'Instancias_APC/breast-cancer_2.arff', 'Instancias_APC/breast-cancer_3.arff', 'Instancias_APC/breast-cancer_4.arff', 'Instancias_APC/breast-cancer_5.arff']
elif num1 == 2:
    fnames = ['Instancias_APC/ecoli_1.arff', 'Instancias_APC/ecoli_2.arff', 'Instancias_APC/ecoli_3.arff', 'Instancias_APC/ecoli_4.arff', 'Instancias_APC/ecoli_5.arff']
elif num1 == 3:
    fnames = ['Instancias_APC/parkinsons_1.arff', 'Instancias_APC/parkinsons_2.arff', 'Instancias_APC/parkinsons_3.arff', 'Instancias_APC/parkinsons_4.arff', 'Instancias_APC/parkinsons_5.arff']
else:
    print("Invalid number entered. Please try again.")

random.seed(seed)
np.random.seed(seed)
# Cambia las etiquetas de los archivos parkinsons de 1.0 y 2.0 a 1 y 2 respectivamente
for i in range(len(fnames)):
    if 'parkinsons' in fnames[i]:
        with open(fnames[i], 'r+') as file:
            lines = file.readlines()
            file.seek(0)
            for line in lines:
                components = line.strip().split(',')
                if components[-1] == '1.0':
                    components[-1] = '1'
                elif components[-1] == '2.0':
                    components[-1] = '2'
                line = ','.join(components) + '\n'
                file.write(line)
            file.truncate()

start_time = time.time()
# Carga los datos de los archivos .arff y los almacena en las listas datos_numpy_list 
# y etiquetas_char_list
for fname in fnames:
    # Carga los datos desde el archivo .arff
    datos, meta = arff.loadarff(fname)
    datos_numpy = np.array([list(d)[:-1] for d in datos], dtype=float)
    etiquetas_char = np.array([d[-1].decode('utf-8') for d in datos], dtype=str)
    datos_numpy_list.append(datos_numpy)
    etiquetas_char_list.append(etiquetas_char)

datos_numpy_concadena = np.concatenate(datos_numpy_list)
etiquetas_char_concadena = np.concatenate(etiquetas_char_list)

# Normalización de los datos
min_vals = np.min(datos_numpy_concadena, axis=0)
max_vals = np.max(datos_numpy_concadena, axis=0)
for i, datos_numpy in enumerate(datos_numpy_list):
    datos_numpy_list[i] = (datos_numpy - min_vals) / (max_vals - min_vals)

datos_numpy_concadena = np.concatenate(datos_numpy_list)

# Convertir las etiquetas a números
le = LabelEncoder()
etiquetas_char_list_encoded = []
for etiquetas_char in etiquetas_char_list:
    etiquetas_char_encoded = le.fit_transform(etiquetas_char)
    etiquetas_char_list_encoded.append(etiquetas_char_encoded)

results = []

vector_instancias = []
for _ in range(50):
    instancia = np.array([np.random.uniform(0, 1, len(datos_numpy_concadena[0]))])
    vector_instancias.append([instancia,None])

if num2 == 1:
    print(f"1. AGG BLX.")
    weights = algoritmo_genetico(vector_instancias, datos_numpy_list, etiquetas_char_list_encoded)
    print(f"AGG BLX -> Peso: {weights}")
    results.append({"Algoritmo": "AGG BLX"})

elif num2 == 2:
    print(f"2. AGG Arit..")
    weights = algoritmo_genetico_ARL(vector_instancias, datos_numpy_list, etiquetas_char_list_encoded)
    print(f"AGG Arit. -> Peso: {weights}")
    results.append({"Algoritmo": "AGG Arit."})

elif num2 == 3:
    print(f"3. AGE BLX.")
    weights = algoritmo_genetico_estacionario(vector_instancias, datos_numpy_list, etiquetas_char_list_encoded)
    print(f"AGE BLX -> Peso: {weights}")
    results.append({"Algoritmo": "AGE BLX"})

elif num2 == 4:
    print(f"4. AGE Arit..")
    weights = algoritmo_genetico_estacionario(vector_instancias, datos_numpy_list, etiquetas_char_list_encoded)
    print(f"AGE Arit. -> Peso: {weights}")
    results.append({"Algoritmo": "AGE Arit."})

elif num2 == 5:
    print(f"5. A. Memético All.")
    weights = memetico_10(vector_instancias, datos_numpy_list, etiquetas_char_list_encoded)
    print(f"AM-All -> Peso: {weights}")
    results.append({"Algoritmo": "AM-All"})

elif num2 == 6:
    print(f"6. A. Memético Rand.")
    weights = memetico_01(vector_instancias, datos_numpy_list, etiquetas_char_list_encoded)
    print(f"AM-Rand -> Peso: {weights}")
    results.append({"Algoritmo": "AM-Rand"})

elif num2 == 7:
    print(f"7. A. Memético Best.")
    weights = memetico_mej(vector_instancias, datos_numpy_list, etiquetas_char_list_encoded)
    print(f"AM-Best -> Peso: {weights}")
    results.append({"Algoritmo": "AM-Best"})

elif num2 == 8:
    print(f"8. BMB.")
    weights = BMB(datos_numpy_list, etiquetas_char_list_encoded)
    print(f"BMB -> Peso: {weights}")
    results.append({"Algoritmo": "BMB"})

elif num2 == 9:
    print(f"9. ILS.")
    weights = ILS(datos_numpy_list, etiquetas_char_list_encoded)
    print(f"ILS -> Peso: {weights}")
    results.append({"Algoritmo": "ILS"})

elif num2 == 10:
    print(f"10. ES.")
    weights = ES(datos_numpy_list, etiquetas_char_list_encoded)
    print(f"ES -> Peso: {weights}")
    results.append({"Algoritmo": "ES"})

elif num2 == 11:
    print(f"11. ILS-ES")
    weights = ILS(datos_numpy_list, etiquetas_char_list_encoded, funcion=ES)
    print(f"ILS-ES -> Peso: {weights}")
    results.append({"Algoritmo": "ILS-ES"})

elif num2 == 12:
    print(f"12. BMB-ES")
    weights = BMB(datos_numpy_list, etiquetas_char_list_encoded, funcion=ES)
    print(f"BMB-ES -> Peso: {weights}")
    results.append({"Algoritmo": "BMB-ES"})

elif num2 == 13:
    print(f"13. ES_modf")
    weights = ES(datos_numpy_list, etiquetas_char_list_encoded,instensidad_mutacion=0.3,vecinos_por_enfriamiento=True)
    print(f"ES_modf -> Peso: {weights}")
    results.append({"Algoritmo": "ES_modf"})

elif num2 == 14:
    print(f"14. BL")
    weights = local_search_P1(datos_numpy_list, etiquetas_char_list_encoded,instancia,max_evaluaciones = 15000)
    print(f"BL -> Peso: {weights}")
    results.append({"Algoritmo": "BL"})

# Cuenta todos los valores de weight que son menores que 0.1
longitud_instancias_menores = np.count_nonzero(weights < 0.1)

# Calcular la tasa de reducción
tasa_reduccion = (longitud_instancias_menores / weights.size)* 100

porcentaje_coincidencia_media = 0
# datos_numpy = test-x etiqueta_char = solution
print("\n*******************************************************************")
print(f"\n¡Datos y etiquetas del archivo {fnames[0]}!")
for i, (datos_numpy, etiquetas_char) in enumerate(zip(datos_numpy_list, etiquetas_char_list_encoded)):
    # métrica o callable, por defecto=-minkowski.
    #Metric para usar para el cálculo a distancia. El predeterminado es "minkowski", que resulta en la distancia euclidiana estándar cuando p = 2. Ver el documentación de scipy.spatial.distance y las métricas enumeradas en distance_metricspara la métrica válida valores.
    clf = KNNClassifier(k=1)
    # Concatenación de datos y etiquetas de otros archivos
    datos_entrenamiento = np.concatenate(
        [datos_numpy_list[(i+1) % 5] for _ in range(4)]
    )
    etiquetas_entrenamiento = np.concatenate(
        [etiquetas_char_list_encoded[(i+1) % 5] for _ in range(4)]
    ) 
    clf.fit(datos_entrenamiento, etiquetas_entrenamiento)
    
    predictions = clf.predict(datos_numpy, weights)
    
    coincidencias = np.sum(predictions == etiquetas_char)
    total = len(predictions)
    porcentaje_coincidencia = (coincidencias / total) * 100
    porcentaje_coincidencia_media = porcentaje_coincidencia + porcentaje_coincidencia_media
    print(f"- {porcentaje_coincidencia:.2f}%")

end_time = time.time()

print(f"Porcentaje de coincidencia promedio: {porcentaje_coincidencia_media / 5:.2f}%")
print(f"Porcentaje de reducción: {tasa_reduccion:.2f}%")
print(f"Fitness: {((0.8 * (porcentaje_coincidencia_media/5)) + (0.2 * tasa_reduccion)):.2f}")
tiempo_ejecucion = end_time - start_time  # Calcula la diferencia, que es el tiempo de ejecución
print(f"La función se ejecutó en {tiempo_ejecucion} segundos")
print("\n*******************************************************************")

results[-1].update({
    "Porcentaje de coincidencia promedio": porcentaje_coincidencia_media / 5,
    "Porcentaje de reduccion": tasa_reduccion,
    "Fitness": ((0.8 * (porcentaje_coincidencia_media / 5)) + (0.2 * tasa_reduccion)),
    "Tiempo de ejecucion": tiempo_ejecucion
})

if not os.path.isfile(name + '.csv'):
    data = pd.DataFrame([results[-1]])
    data.columns = ['Algoritmo', 'Porcentaje de coincidencia promedio', 'Porcentaje de reduccion', 'Fitness', 'Tiempo de ejecucion']
    data.to_csv(name + '.csv', index=False)
else:
    data = pd.read_csv(name + '.csv', encoding='ISO-8859-1')
    df = pd.DataFrame([results[-1]])
    data = pd.concat([data, df], ignore_index=True)
    data.to_csv(name + '.csv', index=False)