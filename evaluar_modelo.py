import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
from tqdm import tqdm

# --- CONFIGURACIÓN DE RUTAS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "dataset")
CSV_PATH = os.path.join(BASE_DIR, "dataset_unificado.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "modelo_cancer_piel_vit.pth")

# --- NUEVO: CARPETA DE ANÁLISIS ---
ANALYSIS_DIR = os.path.join(SCRIPT_DIR, "analisis")
os.makedirs(ANALYSIS_DIR, exist_ok=True) # Crea la carpeta si no existe

BATCH_SIZE = 32

# Mapeo
LABEL_MAP = { 'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6 }
CLASS_NAMES = ['Nevus', 'Melanoma', 'Queratosis B.', 'Carcinoma Basal', 'Queratosis Act.', 'Vascular', 'Dermatofibroma']

# --- CLASE DATASET ---
class SkinTestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        label = LABEL_MAP[row['dx']]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 1. PREPARAR DATOS ---
print("📂 Cargando dataset...")
df = pd.read_csv(CSV_PATH)
df = df[df['dx'].isin(LABEL_MAP.keys())]

# Misma semilla que en el entrenamiento para ser justos
_, val_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)
print(f"🧐 Evaluando {len(val_df)} imágenes de validación.")

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset = SkinTestDataset(val_df, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. CARGAR MODELO ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Dispositivo: {device}")

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=7
)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
except FileNotFoundError:
    print("❌ Error: No encuentro el modelo .pth")
    exit()

# --- 3. INFERENCIA ---
y_true = []
y_pred = []

print("⚡ Generando predicciones...")
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs.logits, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# --- 4. GENERAR REPORTES EN CARPETA 'analisis' ---

# A. Accuracy
acc = accuracy_score(y_true, y_pred)
resumen = f"🏆 Precisión Global (Accuracy): {acc*100:.2f}%\n"
print("\n" + resumen)

# B. Matriz de Confusión (Imagen)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicción de la IA')
plt.ylabel('Realidad')
plt.title(f'Matriz de Confusión - Accuracy: {acc*100:.1f}%')

# GUARDAR IMAGEN EN CARPETA
ruta_imagen = os.path.join(ANALYSIS_DIR, 'matriz_confusion.png')
plt.savefig(ruta_imagen)
print(f"✅ Imagen guardada en: {ruta_imagen}")

# C. Reporte de Texto
reporte = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
print(reporte)

# D. Sensibilidad Melanoma
mel_idx = 1
tp_mel = cm[mel_idx, mel_idx]
fn_mel = sum(cm[mel_idx, :]) - tp_mel
sensibilidad = 0
if (tp_mel + fn_mel) > 0:
    sensibilidad = tp_mel / (tp_mel + fn_mel) * 100

texto_melanoma = f"\n💀 SENSIBILIDAD MELANOMA: {sensibilidad:.2f}%\n"
print(texto_melanoma)

# GUARDAR REPORTE EN TEXTO
ruta_texto = os.path.join(ANALYSIS_DIR, 'reporte_detallado.txt')
with open(ruta_texto, "w", encoding="utf-8") as f:
    f.write("--- REPORTE DE EVALUACIÓN IA DERMA ---\n\n")
    f.write(resumen)
    f.write(texto_melanoma)
    f.write("\n--- DETALLE POR CLASE ---\n")
    f.write(reporte)

print(f"✅ Reporte de texto guardado en: {ruta_texto}")