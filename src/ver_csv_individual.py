import pandas as pd

df = pd.read_csv("../data/dataset.csv")

print(df.head(30)) #primeras filas
print("\nValores únicos en pixels:")

#print("Shape:", df.shape)
#print("Columnas:", df.columns[-5:])  # últimas columnas
#print("Labels únicos:", df["label"].unique())

# verificar valores únicos (sin contar label)
valores = pd.unique(df.drop(columns=["label"]).values.ravel())
print(valores)