import streamlit as st
import torch
import os
import uuid
from datetime import datetime
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="DermaAI - Diagnóstico Inteligente",
    page_icon="🧬",
    layout="centered"
)

# --- RUTAS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "modelo_cancer_piel_vit.pth")
INBOX_DIR = os.path.join(SCRIPT_DIR, "dataset_nuevos_casos")
os.makedirs(INBOX_DIR, exist_ok=True)

# --- DICCIONARIOS MÉDICOS ---
LABEL_MAP = { 0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc', 4: 'akiec', 5: 'vasc', 6: 'df' }

INFO_CLINICA = {
    'nv': {'nombre': 'Nevus (Lunar común)', 'color': 'green', 'icono': '🟢', 'msg': 'Lesión benigna común.'},
    'mel': {'nombre': 'Melanoma', 'color': 'red', 'icono': '🚨', 'msg': '¡ALERTA! Posible lesión maligna. Requiere atención.'},
    'bkl': {'nombre': 'Queratosis Benigna', 'color': 'green', 'icono': '🟢', 'msg': 'Lesión benigna tipo verruga/mancha.'},
    'bcc': {'nombre': 'Carcinoma Basocelular', 'color': 'red', 'icono': '🔴', 'msg': 'Cáncer de piel común. Consultar dermatólogo.'},
    'akiec': {'nombre': 'Queratosis Actínica / Carcinoma', 'color': 'orange', 'icono': '🟠', 'msg': 'Lesión pre-cancerosa o temprana.'},
    'vasc': {'nombre': 'Lesión Vascular', 'color': 'green', 'icono': '🩸', 'msg': 'Benigno. Acumulación de vasos sanguíneos.'},
    'df': {'nombre': 'Dermatofibroma', 'color': 'green', 'icono': '🟤', 'msg': 'Nódulo benigno y firme.'}
}

# --- CARGAR MODELO (CON CACHÉ PARA RAPIDEZ) ---
@st.cache_resource
def cargar_modelo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=7
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None, None

model, device = cargar_modelo()

# Transformación
transformacion = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- INTERFAZ GRÁFICA ---

st.title("🧬 DermaAI")
st.markdown("**Asistente de Dermatología con Inteligencia Artificial**")
st.info("Sube una foto clara de la lesión o usa la cámara.")

# Selector: ¿Cámara o Archivo?
opcion = st.radio("Selecciona método de entrada:", ["📸 Usar Cámara", "📂 Subir Foto"], horizontal=True)

imagen_usuario = None

if opcion == "📸 Usar Cámara":
    imagen_usuario = st.camera_input("Toma una foto de la piel")
else:
    imagen_usuario = st.file_uploader("Sube una imagen (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

# --- LÓGICA DE ANÁLISIS ---
if imagen_usuario is not None and model is not None:
    # Mostrar imagen
    image = Image.open(imagen_usuario).convert("RGB")
    st.image(image, caption="Imagen analizada", use_column_width=True)
    
    with st.spinner('🔬 Analizando células...'):
        # Inferencia
        img_tensor = transformacion(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        confianza, idx = torch.max(probs, 1)
        porcentaje = confianza.item() * 100
        codigo = LABEL_MAP[idx.item()]
        info = INFO_CLINICA[codigo]

    # --- RESULTADOS ---
    st.divider()
    
    # Encabezado del resultado
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"# {info['icono']}")
    with col2:
        st.subheader(f"Diagnóstico: {info['nombre']}")
    
    # Barra de confianza
    st.markdown(f"**Nivel de Confianza IA:** {porcentaje:.1f}%")
    st.progress(int(porcentaje))
    
    # Lógica de Incertidumbre y Alertas
    if porcentaje < 70:
        st.warning("⚠️ **RESULTADO INCIERTO:** La IA no está segura. La foto podría estar borrosa, oscura o muy lejos. Intenta tomarla de nuevo con mejor luz.")
        folder_prefix = "incierto"
    else:
        if info['color'] == 'red':
            st.error(f"🚨 **ACCIÓN RECOMENDADA:** {info['msg']}")
        elif info['color'] == 'orange':
            st.warning(f"⚠️ **ATENCIÓN:** {info['msg']}")
        else:
            st.success(f"✅ **NOTA:** {info['msg']}")
        folder_prefix = codigo

    # --- GUARDAR DATOS (MEMORIA) ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{timestamp}_{folder_prefix}.jpg"
    ruta_guardado = os.path.join(INBOX_DIR, nombre_archivo)
    image.save(ruta_guardado)
    
    # Sidebar con historial
    st.sidebar.write("---")
    st.sidebar.write("📂 **Historial Reciente**")
    st.sidebar.image(image, caption=f"{timestamp} - {codigo}", width=100)
    st.toast("✅ Caso guardado en base de datos para aprendizaje futuro.")

# Footer
st.markdown("---")
st.caption("Nota: Esta herramienta es un apoyo diagnóstico experimental y no sustituye la opinión de un médico profesional.")