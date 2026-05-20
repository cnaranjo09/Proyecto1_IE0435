# Informe Final — Proyecto 1 IE0435  
## Clasificación de contaminaciones en una línea de producción simulada mediante aprendizaje automático clásico

---

# 1. Introducción

En este proyecto se desarrolló un sistema de clasificación automática capaz de detectar contaminaciones en una línea de producción simulada utilizando técnicas de aprendizaje automático clásico. La simulación consistió en una superficie blanca sobre la cual podían aparecer distintos objetos. La presencia de granos de arroz fue considerada una contaminación positiva, mientras que la ausencia de arroz o la presencia de otros objetos como clips o aros fue considerada una clase negativa.

El objetivo principal fue entrenar modelos de clasificación capaces de identificar automáticamente imágenes con presencia de arroz a partir de datos binarios obtenidos desde imágenes procesadas en blanco y negro.

---

# 2. Objetivos

## Objetivo general

Desarrollar un sistema de clasificación basado en aprendizaje automático clásico para detectar contaminaciones en imágenes binarias de una línea de producción simulada.

## Objetivos específicos

- Recolectar y procesar imágenes binarias de tamaño 128×128 píxeles.
- Convertir imágenes en vectores numéricos binarios.
- Construir un dataset etiquetado para clasificación.
- Entrenar y comparar múltiples modelos de aprendizaje automático.
- Seleccionar el modelo con mejor desempeño.
- Exportar el modelo entrenado en formato `.joblib`.
- Garantizar reproducibilidad mediante documentación y control de dependencias.

---

# 3. Recolección de datos

Cada estudiante recolectó:

- 15 imágenes positivas:
  - presencia de arroz
- 15 imágenes negativas:
  - ausencia de arroz
  - presencia de otros objetos

Las imágenes fueron tomadas bajo distintas condiciones de iluminación, posiciones y distribuciones de objetos para aumentar la variabilidad del dataset.

Posteriormente, las imágenes fueron:

- convertidas a blanco y negro
- redimensionadas a 128×128 píxeles
- representadas como matrices binarias

---

# 4. Procesamiento de imágenes

Cada imagen fue transformada en una representación binaria donde:

- `1` representa un píxel blanco
- `0` representa presencia de objeto

Debido a que cada imagen tiene resolución 128×128:

```text
128 × 128 = 16384 píxeles
```

Cada imagen fue convertida a un vector de 16384 características.

Adicionalmente, se agregó una columna de etiqueta:

- `1` → presencia de arroz (positivo)
- `0` → ausencia de arroz (negativo)

Por tanto, cada fila del dataset contiene:

```text
16384 características + 1 etiqueta
```

Total:

```text
16385 columnas
```

---

# 5. Construcción del dataset

El entrenamiento se realizó utilizando los archivos CSV aportados por varios estudiantes del curso.

Cada archivo CSV contenía:

- vectores binarios de imágenes
- una columna final correspondiente a la etiqueta de clasificación

Todos los CSV fueron integrados en un único dataset.

---

# 6. Problemas encontrados durante integración de datos

Durante la unión de los datasets se detectaron varios problemas:

- inconsistencias en nombres de columnas
- columnas adicionales
- presencia de valores `NaN`
- diferencias entre etiquetas

Esto generaba errores durante el entrenamiento.

Para solucionar el problema:

- las columnas fueron alineadas por posición
- se validó que la última columna correspondiera a la etiqueta
- se eliminaron filas inválidas

---

# 7. Modelos utilizados

Se implementaron y compararon los siguientes modelos de clasificación clásica:

## 7.1 K-Nearest Neighbors (KNN)

Modelo basado en similitud entre ejemplos utilizando distancia entre vectores.

### Hiperparámetros evaluados

- número de vecinos:
  - 1
  - 3
  - 5
  - 7

---

## 7.2 Naive Bayes (BernoulliNB)

Modelo probabilístico adecuado para datos binarios.

---

## 7.3 Árbol de decisión

Modelo basado en reglas y divisiones jerárquicas de datos.

### Hiperparámetros evaluados

- profundidad máxima:
  - 5
  - 10
  - 20
  - sin límite

---

## 7.4 Support Vector Machine (SVM)

Modelo que busca maximizar la separación entre clases.

### Hiperparámetros evaluados

- kernel:
  - linear
  - rbf

- parámetro C:
  - 0.1
  - 1
  - 10

---

# 8. División de datos

El dataset fue dividido utilizando:

- 80% entrenamiento
- 20% prueba

Se utilizó:

```python
stratify=y
```

para mantener el balance entre clases.

---

# 9. Validación y selección del modelo

Para optimizar hiperparámetros se utilizó:

```python
GridSearchCV
```

con validación cruzada de 5 particiones (`cv=5`).

El mejor modelo fue seleccionado utilizando el mejor puntaje promedio de validación cruzada.

---

# 10. Métricas de evaluación

Se utilizaron las siguientes métricas:

## Accuracy

Mide la proporción total de predicciones correctas.

## Precision

Mide qué porcentaje de las predicciones positivas fueron correctas.

## Recall

Mide cuántos positivos reales fueron detectados.

## F1-score

Promedio armónico entre precision y recall.

## Support

Cantidad de ejemplos reales de cada clase.

---

# 11. Resultados

Los modelos obtuvieron desempeños altos debido a la naturaleza relativamente simple del problema y al uso de imágenes binarias.

El modelo seleccionado fue exportado en formato:

```text
.joblib
```

## Modelo seleccionado

Después de entrenar y evaluar múltiples modelos de clasificación clásica, el modelo con mejor desempeño general fue:

```text
Support Vector Machine (SVM)
```

El modelo fue seleccionado utilizando validación cruzada (`GridSearchCV`) y obtuvo un score promedio de:

```text
0.7595
```

Además, durante la evaluación sobre el conjunto de prueba se obtuvieron los siguientes resultados:

| Métrica | Valor |
|---|---|
| Accuracy | 0.7917 |
| Precision clase 0 | 1.00 |
| Precision clase 1 | 0.71 |
| Recall clase 0 | 0.58 |
| Recall clase 1 | 1.00 |
| F1-score clase 0 | 0.74 |
| F1-score clase 1 | 0.83 |

El modelo SVM logró detectar correctamente todos los ejemplos positivos del conjunto de prueba (`recall = 1.00` para la clase positiva), aunque presentó algunos falsos positivos en ejemplos negativos.





---

# 12. Inferencia

La inferencia consiste en utilizar el modelo previamente entrenado para clasificar nuevas imágenes no vistas durante el entrenamiento.

El proceso de inferencia incluye:

1. cargar el modelo `.joblib`
2. procesar la imagen
3. convertirla a vector binario
4. realizar predicción
5. mostrar resultado

---

# 13. Limitaciones

El sistema presenta algunas limitaciones:

- sensibilidad a iluminación extrema
- posibles errores con objetos pequeños similares al arroz
- dependencia de fondos blancos
- bajo desempeño ante desenfoque
- dataset relativamente pequeño

---



---

# 15. Conclusiones

Se logró desarrollar exitosamente un sistema de clasificación de contaminaciones utilizando aprendizaje automático clásico.

El proyecto permitió:

- aplicar técnicas de procesamiento de imágenes
- construir datasets binarios
- entrenar múltiples modelos de clasificación
- evaluar desempeño mediante métricas estándar
- implementar inferencia sobre nuevas imágenes

Además, se evidenció la importancia de:

- limpieza de datos
- validación de datasets
- consistencia de formatos
- reproducibilidad experimental

---

---

# 17. Herramientas utilizadas

- Python 3
- NumPy
- Pandas
- Scikit-learn
- Pillow
- Joblib

---

# 18. Estructura del proyecto

```text
Proyecto1_IE0435/
│
├── dataset/
├── data/
│   └── csvs/
├── models/
├── reports/
|   ├── Informe.md
|   ├── MODEL_CARD.md
├── src/
│   ├── train.py
│   ├── exp_matriz.py
|   ├── ver_csv_individual.py
│   └── opencsv.py
├── requirements.txt
├── README.md
├── DATASET.md
└── LICENSE
