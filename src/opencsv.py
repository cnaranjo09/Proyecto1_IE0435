import pandas as pd
import os

EXPECTED_PIXELS = 128 * 128

def validar_csv(path):
    print(f"\n📄 Revisando: {path}")

    df = pd.read_csv(path)

    # -------------------------
    # COLUMNAS
    # -------------------------
    cols = df.columns.tolist()

    # -------------------------
    # 👀 MOSTRAR DATASET
    # -------------------------
    print(f"📦 Número de filas: {df.shape[0]}")
    print(f"🧾 Total columnas: {len(cols)}")
    cols_to_show = cols[:3] + cols[-3:]

    print("\n📊 Vista del dataset (todas las filas, primeras y últimas 3 columnas):")
    print(df[cols_to_show].to_string(index=False))

    print(f"\n🧾 Total columnas: {len(cols)}")

    # -------------------------
    # 🔥 DETECTAR ETIQUETA
    # -------------------------
    possible_label_cols = [
        c for c in cols
        if df[c].dtype == "object" or "label" in c.lower() or "class" in c.lower()
    ]

    if len(possible_label_cols) == 1:
        label_col = possible_label_cols[0]
        pixel_cols = [c for c in cols if c != label_col]
    else:
        label_col = cols[-1]
        pixel_cols = cols[:-1]

    n_pixels = len(pixel_cols)

    print(f"\n🔢 Pixeles detectados: {n_pixels}")

    if n_pixels == EXPECTED_PIXELS:
        print("✔ Pixeles correctos (16384)")
    else:
        print(f"❌ Pixeles incorrectos (esperado {EXPECTED_PIXELS})")

    print(f"🏷 Etiqueta detectada: {label_col}")

    if n_pixels == EXPECTED_PIXELS:
        print("✅ Archivo válido")
    else:
        print("⚠ Archivo NO válido")


folder = "../data/csvs"

for file in os.listdir(folder):
    if file.endswith(".csv"):
        validar_csv(os.path.join(folder, file))

#"../csvs/estudiante3.csv"