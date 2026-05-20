
# MODEL CARD

# Nombre del Modelo

Clasificador de contaminaciones en línea de producción simulada

---

# Version

v1.0

---

# Tipo de Modelo

Clasificación binaria usando aprendizaje automático clásico.

---

# Uso previsto

Detectar presencia de granos de arroz en imágenes binarias de una línea de producción simulada.

---

# Fuera de alcance

Este modelo no fue diseñado para:

- ambientes industriales reales
- imágenes a color
- fondos distintos al blanco
- detección de múltiples tipos de contaminantes
- imágenes borrosas o de baja resolución

---

# Data Summary

El dataset fue construido a partir de imágenes recolectadas por múltiples estudiantes.

Cada estudiante aportó:

- 15 imágenes positivas
- 15 imágenes negativas

Las imágenes fueron:

- convertidas a blanco y negro
- redimensionadas a 128×128 píxeles
- transformadas en vectores binarios

Cada ejemplo contiene:

- 16384 características binarias
- 1 etiqueta de clasificación

---

# Proceso de etiquetado

Etiquetas utilizadas:

- `1` → presencia de arroz
- `0` → ausencia de arroz

La etiqueta corresponde a la última columna de cada archivo CSV.

---

# Metricas

Se utilizaron:

- Accuracy
- Precision
- Recall
- F1-score

La evaluación se realizó utilizando:

- train/test split 80/20
- validación cruzada de 5 particiones

---

# Modelos evaluados

- KNN
- Naive Bayes
- Decision Tree
- SVM

---

# Modelo seleccionada

El mejor modelo fue seleccionado utilizando el mejor score promedio de validación cruzada.

---

# Consideraciones éticas y de seguridad

El modelo puede presentar sesgos debido a:

- iluminación
- resolución de cámara
- variaciones del fondo
- calidad de imágenes

No debe utilizarse en sistemas críticos reales sin validación adicional.

---

# Limitaciones

- Sensible a desenfoque
- Dependencia de fondo blanco
- Dataset pequeño
- Posibles falsos positivos con objetos similares al arroz

---

# Reproducibility

## Entrenamiento

Para el entrenamiento se utiliza el archivo:

```bash
python train.py
```
