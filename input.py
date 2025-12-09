import pandas as pd
import numpy as np

# --- Configuración ---
input_path = "HR_dataset_400.csv"   # ruta del archivo original
output_path = "HR_dataset_400_E.csv"    # ruta del archivo resultante
n_new = 400                                 # número de filas a agregar
random_state = 42                           # semilla para reproducibilidad
replace_if_needed = True                    # permitir muestreo con reemplazo si hay < n_new

# --- Cargar dataset ---
df = pd.read_csv(input_path)

# --- Detectar columna de attrition (tolerante a mayúsculas y errores comunes) ---
possible_names = ["attrition", "Attrition", "atrittion", "ATRICTION", "ATRITION"]
attr_col = None
for name in possible_names:
    if name in df.columns:
        attr_col = name
        break
# Si no se encontró, intenta buscar por substring
if attr_col is None:
    for col in df.columns:
        if "attr" in col.lower():
            attr_col = col
            break

if attr_col is None:
    raise ValueError("No se encontró una columna de attrition en el dataset. Revisa los nombres de columnas.")

# --- Filtrar filas con attrition == 1 ---
# Aseguramos que la comparación funcione con valores numéricos o strings '1'/'Yes'/'True'
mask = None
if pd.api.types.is_numeric_dtype(df[attr_col]):
    mask = df[attr_col] == 0
else:
    # normalizar a minúsculas strings para comparar
    mask = df[attr_col].astype(str).str.lower().isin(["1", "true", "yes", "y", "si", "sí"])

df_attr1 = df[mask].copy()
n_available = len(df_attr1)
if n_available == 0:
    raise ValueError("No hay filas con attrition == 1 en el dataset.")

# --- Muestreo ---
replace = replace_if_needed and (n_available < n_new)
sampled = df_attr1.sample(n=n_new, replace=replace, random_state=random_state).copy()

# --- Opcional: ajustar identificador único si existe (ej. EmployeeID) ---
# Si hay una columna que parezca ID, incrementamos para evitar duplicados exactos
id_candidates = [c for c in df.columns if "id" in c.lower()]
if id_candidates:
    id_col = id_candidates[0]  # tomar la primera candidata
    if pd.api.types.is_integer_dtype(df[id_col]) or pd.api.types.is_float_dtype(df[id_col]):
        max_id = int(df[id_col].max(skipna=True))
        # reasignar IDs nuevos consecutivos a las filas muestreadas
        sampled[id_col] = range(max_id + 1, max_id + 1 + len(sampled))

# --- Concatenar y reindexar ---
df_augmented = pd.concat([df, sampled], ignore_index=True)
df_augmented.reset_index(drop=True, inplace=True)

# --- Guardar resultado ---
df_augmented.to_csv(output_path, index=False)
print(f"Original: {len(df)} filas. Agregadas: {len(sampled)} filas. Total: {len(df_augmented)} filas.")
print(f"Archivo guardado en: {output_path}")