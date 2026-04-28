# Proyecto1_IE0435
Proyecto sobre algoritmos de IA

## Reproducir el entrenamiento del modelo

### 1. Clonar el repositorio

```bash
git clone <https://github.com/cnaranjo09/Proyecto1_IE0435.git>
cd <Proyecto1_IE0435>
```
### 2. Crear entorno virtual
```bash
python3 -m venv venv
```
### 3. Activar entorno virtual
### Linux
```bash
source venv/bin/activate
```
### Windows:
venv\Scripts\activate

### 4. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 5. Verificar estructura del proyecto (IMPORTANTE)
 Debe existir:
 dataset/
   ├── positivos/
   └── negativos/

### 6. Ejecutar entrenamiento
```
cd src
python train.py
```

### 7. Resultados esperados
 - Se genera el modelo entrenado en:
   [models/B44870_Carlos_Naranjo.joblib](https://github.com/cnaranjo09/Proyecto1_IE0435/tree/main/models)
   
- Se genera el dataset en formato CSV:
   [data/dataset.csv](https://github.com/cnaranjo09/Proyecto1_IE0435/tree/main/data)

 - Se muestran métricas en consola (accuracy, precision, recall, f1-score)

