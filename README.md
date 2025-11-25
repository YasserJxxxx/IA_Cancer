1. Instalación de Dependencias (Actualizado)
Abre tu terminal (cmd o PowerShell) y asegúrate de tener todo esto instalado. (Si ya arreglaste lo de dlib, solo te faltará gdown).

Bash

# 1. Herramientas para compilar (cmake y dlib - ya deberías tenerlas)
pip install cmake
pip install dlib

# 2. Bibliotecas del proyecto (OpenCV, face_recognition)
pip install opencv-python
pip install face_recognition
pip install numpy

# 3. NUEVA: Biblioteca para descargar de Google Drive
pip install gdown
2. Estructura de Carpetas
Solo necesitas tener los dos scripts en la misma carpeta. El script de creación descargará las fotos automáticamente.

Reconocimiento Facial/
├── crear_modelo.py
└── reconocimiento_en_vivo.py
3. Script 1: crear_modelo.py (Actualizado con Google Drive)
Este script ahora se conectará a tu enlace de Google Drive, descargará las fotos a una carpeta temporal (dataset_descargado) y luego creará el modelo_caras.pkl.

Python

import face_recognition
import os
import pickle
import cv2
import gdown  # Biblioteca para descargar de Google Drive
import shutil # Para borrar la carpeta temporal al final

print("--- [PASO 1] Descargando dataset desde Google Drive... ---")

# ID de tu carpeta de Google Drive
FOLDER_ID = "1iNVfq-7bNH3QP-RSOpTQ5D9sU5bi-mFx"
CARPETA_DATASET_LOCAL = "dataset_descargado" 

# Crear la carpeta local si no existe
if not os.path.exists(CARPETA_DATASET_LOCAL):
    os.makedirs(CARPETA_DATASET_LOCAL)

# Descargamos el contenido de la carpeta pública de Drive
try:
    gdown.download_folder(id=FOLDER_ID, output=CARPETA_DATASET_LOCAL, quiet=False, use_cookies=False)
    print("\nDescarga del dataset completada.")
except Exception as e:
    print(f"\n[ERROR FATAL] No se pudo descargar la carpeta de Google Drive.")
    print("Asegúrate de que el enlace es correcto y que la carpeta es pública ('Cualquier persona con el enlace puede ver').")
    print(f"Detalle: {e}")
    exit()

# --- [PASO 2] Procesamiento de Imágenes y Creación de Embeddings (CNN) ---

print("\n--- [PASO 2] Procesando imágenes y generando modelo... ---")
embeddings_conocidos = []
nombres_conocidos = []

# Recorremos la carpeta local que acabamos de descargar
for nombre_archivo in os.listdir(CARPETA_DATASET_LOCAL):
    ruta_completa = os.path.join(CARPETA_DATASET_LOCAL, nombre_archivo)
    
    if os.path.isdir(ruta_completa):
        continue

    # Extraer el nombre (ej. "ana_perez.jpg" -> "Ana Perez")
    nombre = os.path.splitext(nombre_archivo)[0].replace("_", " ").title()
    print(f"Procesando a: {nombre}")

    try:
        imagen = face_recognition.load_image_file(ruta_completa)
        # Usamos la CNN para obtener el embedding (vector de 128D)
        embedding = face_recognition.face_encodings(imagen)[0]
        
        embeddings_conocidos.append(embedding)
        nombres_conocidos.append(nombre)
        
    except IndexError:
        print(f"  [AVISO] No se detectó ninguna cara en {nombre_archivo}. Saltando.")
    except Exception as e:
        print(f"  [ERROR] No se pudo procesar {nombre_archivo}. Error: {e}")

# --- [PASO 3] Guardado del Modelo .pkl ---
datos_modelo = {"embeddings": embeddings_conocidos, "nombres": nombres_conocidos}
NOMBRE_MODELO = "modelo_caras.pkl"

with open(NOMBRE_MODELO, "wb") as f:
    pickle.dump(datos_modelo, f)

# --- [PASO 4] Limpieza (Opcional) ---
try:
    print(f"\n--- [PASO 4] Limpiando carpeta de dataset temporal ({CARPETA_DATASET_LOCAL})... ---")
    shutil.rmtree(CARPETA_DATASET_LOCAL)
    print("Limpieza completada.")
except Exception as e:
    print(f"No se pudo borrar la carpeta temporal: {e}")

print(f"\n¡'Modelo' guardado exitosamente como '{NOMBRE_MODELO}'!")
print(f"Se procesaron {len(nombres_conocidos)} integrantes.")
4. Script 2: reconocimiento_en_vivo.py (Sin cambios)
Este script no necesita cambios. Simplemente carga el modelo_caras.pkl que generó el script anterior y enciende la cámara.

Python

import face_recognition
import cv2
import pickle
import numpy as np
import os

print("Cargando modelo de reconocimiento facial...")
NOMBRE_MODELO = "modelo_caras.pkl"

# --- 1. Carga del Modelo Guardado ---
if not os.path.exists(NOMBRE_MODELO):
    print(f"[ERROR] No se encuentra el archivo del modelo: {NOMBRE_MODELO}")
    print("Por favor, ejecuta primero 'crear_modelo.py' para generarlo.")
    exit()

with open(NOMBRE_MODELO, "rb") as f:
    datos_modelo = pickle.load(f)

embeddings_conocidos = datos_modelo["embeddings"]
nombres_conocidos = datos_modelo["nombres"]

print(f"Modelo cargado. {len(nombres_conocidos)} integrantes conocidos.")

# --- 2. Iniciar Cámara y Reconocimiento en Vivo ---
print("Iniciando cámara... (Presiona 'q' en la ventana de video para salir)")
video_capture = cv2.VideoCapture(0) # 0 es la cámara por defecto

if not video_capture.isOpened():
    print("[ERROR] No se pudo abrir la cámara.")
    exit()

# Factor para escalar el frame y procesar más rápido (opcional)
FACTOR_ESCALADO = 0.5 

while True:
    # Capturar un solo frame de video
    ret, frame = video_capture.read()
    if not ret:
        print("Error al capturar frame.")
        break

    # Escalar el frame para un procesamiento más rápido
    frame_pequeno = cv2.resize(frame, (0, 0), fx=FACTOR_ESCALADO, fy=FACTOR_ESCALADO)
    rgb_frame_pequeno = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)

    # --- 3. Detección y Reconocimiento (CNNs en acción) ---
    locs_caras = face_recognition.face_locations(rgb_frame_pequeno)
    embeddings_caras_actuales = face_recognition.face_encodings(rgb_frame_pequeno, locs_caras)

    nombres_en_frame = []
    for embedding_cara in embeddings_caras_actuales:
        coincidencias = face_recognition.compare_faces(embeddings_conocidos, embedding_cara, tolerance=0.5)
        nombre = "Desconocido"

        distancias = face_recognition.face_distance(embeddings_conocidos, embedding_cara)
        
        if len(distancias) > 0:
            mejor_coincidencia_idx = np.argmin(distancias)
            if coincidencias[mejor_coincidencia_idx]:
                nombre = nombres_conocidos[mejor_coincidencia_idx]

        nombres_en_frame.append(nombre)

    # --- 4. Visualización (Dibujar los resultados) ---
    factor_inverso = 1 / FACTOR_ESCALADO

    for (top, right, bottom, left), nombre in zip(locs_caras, nombres_en_frame):
        # Escalar las coordenadas de vuelta al tamaño original
        top = int(top * factor_inverso)
        right = int(right * factor_inverso)
        bottom = int(bottom * factor_inverso)
        left = int(left * factor_inverso)

        # Dibujar un rectángulo alrededor de la cara
        color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Dibujar una etiqueta con el nombre debajo
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, nombre, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Mostrar la imagen resultante
    cv2.imshow('Reconocimiento Facial (Presiona "q" para salir)', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. Limpieza ---
video_capture.release()
cv2.destroyAllWindows()
print("Cerrando aplicación.")
