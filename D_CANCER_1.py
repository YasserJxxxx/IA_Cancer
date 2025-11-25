import os
import pandas as pd
from glob import glob
import sys

# --- 1. CONFIGURACIÓN DE RUTAS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "dataset")
DIR_HAM = os.path.join(BASE_DIR, "extracted_ham10000")
DIR_PAD = os.path.join(BASE_DIR, "extracted_pad_ufes")
OUTPUT_CSV = os.path.join(BASE_DIR, "dataset_unificado.csv")

print(f"📂 Trabajando en: {BASE_DIR}")

# --- FUNCIÓN DE BÚSQUEDA MEJORADA ---
def buscar_imagenes_recursivamente(directorio):
    if not os.path.exists(directorio):
        return {}
    
    print(f"   🔎 Escaneando: {os.path.basename(directorio)}...")
    patrones = ['*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG']
    rutas = []
    for p in patrones:
        rutas.extend(glob(os.path.join(directorio, '**', p), recursive=True))
    
    # Creamos diccionario { ID_LIMPIO : RUTA_COMPLETA }
    # ID_LIMPIO = Nombre del archivo sin extensión (ej: 'img1' de 'img1.png')
    mapa = {os.path.splitext(os.path.basename(x))[0]: x for x in rutas}
    
    # DEBUG: Mostrar qué encontró
    if len(mapa) > 0:
        ejemplos = list(mapa.keys())[:3]
        print(f"      👀 Ejemplos de archivos encontrados en carpeta: {ejemplos}")
    
    return mapa

def encontrar_csv(directorio):
    csvs = glob(os.path.join(directorio, '**', '*.csv'), recursive=True)
    for c in csvs:
        if 'metadata' in os.path.basename(c).lower(): return c
    return csvs[0] if csvs else None

# --- PROCESO ---

# 1. HAM10000
print("\n🔄 [1/2] HAM10000...")
mapa_ham = buscar_imagenes_recursivamente(DIR_HAM)
csv_ham = encontrar_csv(DIR_HAM)

if csv_ham and mapa_ham:
    df_ham = pd.read_csv(csv_ham)
    df_ham['path'] = df_ham['image_id'].map(mapa_ham.get)
    df_ham['source'] = 'HAM10000'
    df_ham = df_ham.dropna(subset=['path'])
    print(f"   ✅ OK: {len(df_ham)} imágenes.")
else:
    df_ham = pd.DataFrame()

# 2. PAD-UFES-20 (AQUÍ ESTABA EL ERROR)
print("\n🔄 [2/2] PAD-UFES-20...")
mapa_pad = buscar_imagenes_recursivamente(DIR_PAD)
csv_pad = encontrar_csv(DIR_PAD)

if csv_pad and mapa_pad:
    print(f"   📄 Leyendo CSV: {os.path.basename(csv_pad)}")
    df_pad = pd.read_csv(csv_pad)
    
    # Detectar columna ID
    col_id = 'img_id' if 'img_id' in df_pad.columns else 'image_id'
    
    # DEBUG: Mostrar qué tiene el CSV antes de limpiar
    print(f"      👀 Ejemplos de IDs en el CSV (Original): {df_pad[col_id].head(3).tolist()}")

    # --- CORRECCIÓN CLAVE ---
    # Forzamos que los IDs del CSV sean strings y les quitamos la extensión (.png)
    # Esto asegura que 'IM_001.png' se convierta en 'IM_001' para coincidir con la carpeta
    df_pad['id_limpio'] = df_pad[col_id].astype(str).apply(lambda x: os.path.splitext(x)[0])
    
    print(f"      👀 Ejemplos de IDs en el CSV (Limpios) : {df_pad['id_limpio'].head(3).tolist()}")

    # Ahora hacemos el match usando el ID LIMPIO
    df_pad['path'] = df_pad['id_limpio'].map(mapa_pad.get)
    
    # Traducción de etiquetas
    traductor = {'BCC': 'bcc', 'MEL': 'mel', 'NEV': 'nv', 'ACK': 'akiec', 'SEK': 'bkl', 'BOD': 'akiec', 'SCC': 'akiec'}
    if 'diagnostic' in df_pad.columns:
        df_pad['dx'] = df_pad['diagnostic'].map(traductor)
    else:
        df_pad['dx'] = None
        
    df_pad['source'] = 'PAD-UFES'
    df_pad = df_pad.rename(columns={col_id: 'image_id'})
    
    # Verificar cuántas cruzaron bien
    total_imgs = len(df_pad)
    con_path = df_pad['path'].notnull().sum()
    print(f"      📊 Coincidencias: {con_path} de {total_imgs} registros del CSV tienen imagen.")
    
    df_pad = df_pad.dropna(subset=['path', 'dx'])
    df_pad = df_pad[['image_id', 'dx', 'path', 'source']]
    print(f"   ✅ OK: {len(df_pad)} imágenes listas.")
else:
    print("   ❌ Fallo crítico en PAD-UFES.")
    df_pad = pd.DataFrame()

# 3. UNIR
print("\n⚗️ Fusionando...")
if not df_ham.empty or not df_pad.empty:
    df_final = pd.concat([df_ham, df_pad], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"🎉 GUARDADO: {OUTPUT_CSV}")
    print(f"Total imágenes: {len(df_final)}")
else:
    print("❌ Error: No hay datos.")