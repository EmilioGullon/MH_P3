import numpy as np
from scipy.spatial import distance
class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, weight): 
        weighted_X = X * weight
        weighted_X_train = self.X_train * weight
        dist = np.array(distance.cdist(weighted_X, weighted_X_train))
        predictions = self.y_train[dist.argmin(axis=1)]
        return predictions
        #     # Calculamos las distancias euclidianas
        #     # cdist es más eficiente que np.linalg.norm para calcular distancias entre matrices
        #     distances = np.linalg.norm(self.X_train - x, axis=1)
        #     # Obtenemos los índices de los k vecinos más cercanos
        #     nearest_neighbors = distances.argsort()[:self.k]
        #     # Votamos por la clase mayoritaria entre los vecinos más cercanos
        #     labels = self.y_train[nearest_neighbors]
            
        #     # Contamos las ocurrencias de cada etiqueta
        #     unique_labels, counts = np.unique(labels, return_counts=True)
        #     # Obtenemos la etiqueta más frecuente
        #     prediction = unique_labels[np.argmax(counts)]
        #     predictions.append(prediction)
        # return np.array(predictions)
