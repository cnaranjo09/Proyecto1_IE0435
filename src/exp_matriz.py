import os
import pandas as pd
import numpy as np
import joblib

from utils import procesar_imagen

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# =========================
# 1. CARGAR DATASET
# =========================


def cargar_dataset():
    X, y = [], []

    print("Procesando positivos...")
    for archivo in os.listdir("../dataset/positivos"):
        ruta = os.path.join("../dataset/positivos", archivo)
        try:
            X.append(procesar_imagen(ruta))
            y.append(1)
            print(f"✔ {archivo}")
        except Exception as e:
            print(f"❌ ERROR en {archivo}: {e}")

    print("\nProcesando negativos...")
    for archivo in os.listdir("../dataset/negativos"):
        ruta = os.path.join("../dataset/negativos", archivo)
        try:
            X.append(procesar_imagen(ruta))
            y.append(0)
            print(f"✔ {archivo}")
        except Exception as e:
            print(f"❌ ERROR en {archivo}: {e}")

    print(f"\nTotal cargadas correctamente: {len(X)}")

    return np.array(X), np.array(y)



def proceso():
    print("Cargando dataset...")
    X, y = cargar_dataset() # llama a la funcion de cargar imagenes
    print("ANTES DE EXPORTAR CSV")
    exportar_csv(X, y)
    


def exportar_csv(X, y):

    print("\nExportando dataset a CSV...")

    # Crear nombres de columnas
    columnas = [f"pixel_{i}" for i in range(X.shape[1])]
    
    # Crear DataFrame
    df = pd.DataFrame(X, columns=columnas)
    
    # Agregar etiqueta
    df["label"] = y

    # Crear carpeta si no existe
    os.makedirs("../data", exist_ok=True)

    # Guardar CSV
    ruta_csv = "../data/dataset.csv"
    df.to_csv(ruta_csv, index=False)

    print(f"CSV guardado en: {ruta_csv}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    proceso()