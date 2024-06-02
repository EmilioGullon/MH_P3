import subprocess
import datetime
import os

# Definir las entradas
entradas = [(i, j) for i in range(1, 14) for j in range(1, 4)]

# Obtener la fecha actual
fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d")

# Crear una carpeta para los resultados
os.makedirs(f"resultados_{fecha_actual}", exist_ok=True)

# Ejecutar el archivo Main.py con cada entrada y redirigir la salida a un archivo
for entrada in entradas:
    # Nombre del archivo de salida con la fecha del día y el número de entrada
    nombre_archivo = f"resultados_{fecha_actual}/salida_{entrada[0]}_{fecha_actual}.txt"
    comando = f"python Main.py {entrada[1]} {entrada[0]} {"comparacion_"+ entrada[1].__str__() +"_del_" + fecha_actual}"
    print("Ejecutando:", comando)
    salida = subprocess.check_output(comando, shell=True, text=True)
    with open(nombre_archivo, "a") as archivo:
        archivo.write(salida)
    print("Comando terminado y copiado")

print("La salida se ha guardado en la carpeta", f"resultados_{fecha_actual}")