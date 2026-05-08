import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# =========================
# 1. CARGAR DATASET DESDE CSVs
# =========================
def cargar_dataset_csvs():
    ruta_csvs = "../data/csvs"
    dataframes = []

    print("Cargando CSVs...")

    for archivo in os.listdir(ruta_csvs):
        if archivo.endswith(".csv"):
            ruta = os.path.join(ruta_csvs, archivo)
            print(f"  -> {archivo}")

            df = pd.read_csv(ruta)

            # 🔥 ignorar nombres de columnas
            df.columns = range(df.shape[1])

            dataframes.append(df)

    # Unir todos
    df_total = pd.concat(dataframes, ignore_index=True)

    print(f"\nTotal de ejemplos: {df_total.shape[0]}")
    print(f"Total de columnas: {df_total.shape[1]}")

    # 🔥 separar por posición
    X = df_total.iloc[:, :-1].values
    y = df_total.iloc[:, -1].values

    # 🔥 limpiar NaN
    mask = ~pd.isna(y)
    X = X[mask]
    y = y[mask]

    print("Valores únicos en y:", set(y))

    return X, y


# =========================
# 2. EVALUAR MODELO
# =========================
def evaluar(modelo, X_test, y_test, nombre):
    y_pred = modelo.predict(X_test)

    print("\n=========================")
    print(f"Modelo: {nombre}")
    print("=========================")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# =========================
# 3. ENTRENAMIENTO
# =========================
def entrenar():
    print("Cargando dataset desde CSVs...")
    X, y = cargar_dataset_csvs()

    print("\nDividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    mejores_modelos = {}

    # =========================
    # KNN
    # =========================
    print("\nEntrenando KNN...")
    knn_params = {'n_neighbors': [1, 3, 5, 7]}

    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
    knn_grid.fit(X_train, y_train)

    knn_best = knn_grid.best_estimator_
    evaluar(knn_best, X_test, y_test, "KNN")

    mejores_modelos["KNN"] = (knn_best, knn_grid.best_score_)

    # =========================
    # Naive Bayes
    # =========================
    print("\nEntrenando Naive Bayes...")
    nb = BernoulliNB()
    nb.fit(X_train, y_train)

    evaluar(nb, X_test, y_test, "Naive Bayes")

    mejores_modelos["NB"] = (nb, nb.score(X_train, y_train))

    # =========================
    # Árbol de decisión
    # =========================
    print("\nEntrenando Árbol de Decisión...")
    tree_params = {'max_depth': [5, 10, 20, None]}

    tree_grid = GridSearchCV(DecisionTreeClassifier(), tree_params, cv=5)
    tree_grid.fit(X_train, y_train)

    tree_best = tree_grid.best_estimator_
    evaluar(tree_best, X_test, y_test, "Árbol")

    mejores_modelos["TREE"] = (tree_best, tree_grid.best_score_)

    # =========================
    # SVM
    # =========================
    print("\nEntrenando SVM...")
    svm_params = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10]
    }

    svm_grid = GridSearchCV(SVC(), svm_params, cv=5)
    svm_grid.fit(X_train, y_train)

    svm_best = svm_grid.best_estimator_
    evaluar(svm_best, X_test, y_test, "SVM")

    mejores_modelos["SVM"] = (svm_best, svm_grid.best_score_)

    # =========================
    # 4. SELECCIONAR MEJOR MODELO
    # =========================
    print("\nSeleccionando mejor modelo...")

    mejor_nombre = None
    mejor_score = -1
    mejor_modelo = None

    for nombre, (modelo, score) in mejores_modelos.items():
        if score > mejor_score:
            mejor_score = score
            mejor_nombre = nombre
            mejor_modelo = modelo

    print(f"\nMejor modelo: {mejor_nombre}")
    print(f"Score (CV): {mejor_score}")

    # =========================
    # 5. GUARDAR MODELO
    # =========================
    os.makedirs("../models", exist_ok=True)

    ruta_modelo = "../models/B44870_Carlos_Naranjo.joblib"
    joblib.dump(mejor_modelo, ruta_modelo)

    print(f"\nModelo guardado en: {ruta_modelo}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    entrenar()
