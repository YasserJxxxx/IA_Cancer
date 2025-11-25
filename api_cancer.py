import os
import io
import uuid
import torch
import shutil
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

# --- CONFIGURACIÓN ---
app = FastAPI(
    title="API Diagnóstico Dermatológico IA",
    description="Backend para detección de cáncer de piel usando Vision Transformers.",
    version="1.0"
)

# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "modelo_cancer_piel_vit.pth")
INBOX_DIR = os.path.join(SCRIPT_DIR, "dataset_nuevos_casos") # Aquí se guardarán las fotos nuevas

# Crear carpeta de "Memoria" si no existe
os.makedirs(INBOX_DIR, exist_ok=True)

# Mapeo de Diagnósticos
LABEL_MAP = { 0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc', 4: 'akiec', 5: 'vasc', 6: 'df' }

INFO_CLINICA = {
    'nv': {
        'nombre': 'Nevus (Lunar común)',
        'gravedad': '🟢 Benigno',
        'accion': 'Observación rutinaria. Si cambia de forma, consultar.'
    },
    'mel': {
        'nombre': 'Melanoma',
        'gravedad': '🔴 ALTA PRIORIDAD',
        'accion': 'ACUDIR A DERMATÓLOGO INMEDIATAMENTE. Requiere biopsia.'
    },
    'bkl': {
        'nombre': 'Queratosis Benigna',
        'gravedad': '🟢 Benigno',
        'accion': 'Generalmente no requiere tratamiento, salvo por estética.'
    },
    'bcc': {
        'nombre': 'Carcinoma Basocelular',
        'gravedad': '🔴 Maligno (Cáncer)',
        'accion': 'Consultar especialista. Crecimiento lento pero destructivo.'
    },
    'akiec': {
        'nombre': 'Queratosis Actínica / Carcinoma',
        'gravedad': '🟠 Pre-cancerígeno / Maligno',
        'accion': 'Tratamiento recomendado para evitar progresión.'
    },
    'vasc': {
        'nombre': 'Lesión Vascular',
        'gravedad': '🟢 Benigno',
        'accion': 'Control rutinario.'
    },
    'df': {
        'nombre': 'Dermatofibroma',
        'gravedad': '🟢 Benigno',
        'accion': 'Inofensivo. Es como una cicatriz interna.'
    }
}

# --- CARGA DEL CEREBRO (MODELO) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Iniciando servidor en: {device}")

try:
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=7
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    print(f"❌ ERROR CRÍTICO AL CARGAR MODELO: {e}")
    # No detenemos el server para que puedas ver el error en el log, 
    # pero las predicciones fallarán.

# Transformación de imagen
transformacion = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- ENDPOINTS (LAS FUNCIONES DE LA API) ---

@app.get("/")
def home():
    return {"estado": "online", "mensaje": "IA Dermatológica lista. Usa /docs para probar."}

@app.post("/analizar/")
async def analizar_lesion(file: UploadFile = File(...), tipo_foto: str = "celular"):
    """
    Recibe una imagen y devuelve el diagnóstico.
    - file: La foto (jpg, png).
    - tipo_foto: 'celular' o 'dermatoscopio' (opcional, para guardar metadata).
    """
    
    # 1. Validar que sea imagen
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.")

    # 2. Leer la imagen
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Archivo de imagen corrupto.")

    # 3. Inferencia (Pensar)
    img_tensor = transformacion(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    # 4. Resultados Matemáticos
    confianza, idx = torch.max(probs, 1)
    porcentaje = confianza.item() * 100
    codigo = LABEL_MAP[idx.item()]
    info = INFO_CLINICA[codigo]

    # --- LÓGICA DE NEGOCIO Y SEGURIDAD ---
    
    respuesta = {
        "diagnostico_tecnico": info['nombre'],
        "gravedad": info['gravedad'],
        "confianza": round(porcentaje, 2),
        "recomendacion": info['accion'],
        "mensaje_alerta": None
    }

    # REGLA 1: Incertidumbre (Si la IA duda, pide ayuda al usuario)
    if porcentaje < 70:
        respuesta["diagnostico_tecnico"] = "No concluyente"
        respuesta["gravedad"] = "⚠️ Incertidumbre alta"
        respuesta["recomendacion"] = "NO SE PUEDE DETERMINAR DIAGNÓSTICO."
        respuesta["mensaje_alerta"] = (
            "La imagen no tiene suficiente calidad o claridad para la IA. "
            "Por favor: 1. Use flash o mejore la luz. "
            "2. Enfoque bien la lesión. "
            "3. Intente tomar la foto un poco más cerca."
        )
    
    # REGLA 2: Guardar datos para Aprendizaje Continuo
    # Guardamos la foto con el diagnóstico que dio la IA para revisarla luego
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    id_unico = str(uuid.uuid4())[:8]
    nombre_archivo = f"{timestamp}_{codigo}_{id_unico}.jpg"
    ruta_guardado = os.path.join(INBOX_DIR, nombre_archivo)
    
    image.save(ruta_guardado)
    # Aquí ya tenemos la foto guardada. En el futuro, un script leerá esta carpeta
    # para re-entrenar el modelo.
    
    respuesta["id_seguimiento"] = nombre_archivo # Para que la App sepa cuál foto fue

    return JSONResponse(content=respuesta)