import joblib
from utils import procesar_imagen

modelo = joblib.load("models/tu_modelo.joblib")

imagen = procesar_imagen("test.png")
pred = modelo.predict([imagen])

print("Resultado:", pred)