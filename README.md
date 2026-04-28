# Proyecto1_IE0435
Proyecto sobre clasificación de contaminaciones usando aprendizaje automático.

## 🔧 Reproducir el entrenamiento del modelo

### 1. Clonar el repositorio

```bash
git clone https://github.com/cnaranjo09/Proyecto1_IE0435.git
cd Proyecto1_IE0435
```

### 2. Requisitos

Instalar python (en caso de no tenerlo)

 **Linux (Ubuntu/Debian)*** 
```bash
sudo apt update 
sudo apt install python3 python3-venv python3-pip -y 
python3 --version
```



### 3. Crear entorno virtual
Se debe crear un entorno virtual porque permite aislar las dependencias del proyecto del resto del sistema, evitando conflictos entre versiones de librerías y asegurando que cualquier persona pueda reproducir el entrenamiento en las mismas condiciones en que fue desarrollado. De esta forma, el proyecto se vuelve más estable y portátil, ya que no depende de configuraciones globales del sistema.

```bash
python3 -m venv venv
```
### 4. Activar entorno virtual
### Linux
```bash
source venv/bin/activate
```

### Windows:
```bash
venv\Scripts\activate
```
### Instalar dependencias
Dentro del entorno virtual
```bash
pip install numpy pandas scikit-learn pillow joblib
```

### 5. Verificar estructura del proyecto (IMPORTANTE)
 Debe existir:
 
```bash
 dataset/
   ├── positivos/
   └── negativos/
```
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

