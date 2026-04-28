import numpy as np
from PIL import Image

def procesar_imagen(ruta):
    img = Image.open(ruta).convert('L')
    img = img.resize((128, 128))
    
    img_array = np.array(img)
    binaria = (img_array > 200).astype(int)
    
    return binaria.flatten()