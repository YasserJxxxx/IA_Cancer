import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm # Para barra de progreso

# --- CONFIGURACIÓN ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "dataset")
CSV_PATH = os.path.join(BASE_DIR, "dataset_unificado.csv")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "modelo_cancer_piel_vit.pth")

# Hyperparámetros (Configuración del cerebro)
BATCH_SIZE = 16       # Cuantas fotos ve a la vez
EPOCHS = 5            # Cuantas veces repasa todo el dataset (Aumentar a 10-20 para prod)
LEARNING_RATE = 2e-5  # Velocidad de aprendizaje (ViT necesita ser lento y preciso)
NUM_CLASSES = 7       # Los 7 tipos de lesiones

# Mapa de etiquetas (Tu IA hablará en números, nosotros traducimos)
LABEL_MAP = {
    'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6
}
# Mapa inverso para cuando la IA responda
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# --- CLASE DEL DATASET ---
class SkinCancerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        label_str = row['dx']
        
        # Cargar imagen y convertir a RGB (ignora transparencia de PNG)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")
            # Si falla, devolvemos una imagen negra para no romper el loop
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        
        # Convertir etiqueta texto a numero
        label = torch.tensor(LABEL_MAP[label_str], dtype=torch.long)
        
        return image, label

# --- PREPARACIÓN DE DATOS ---
print("📂 Cargando dataset unificado...")
df = pd.read_csv(CSV_PATH)

# Filtramos clases raras si las hubiera
df = df[df['dx'].isin(LABEL_MAP.keys())]

# División: 80% Entrenar (Aprender), 20% Validar (Examen)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)

print(f"📊 Entrenamiento: {len(train_df)} imágenes")
print(f"📊 Validación:    {len(val_df)} imágenes")

# Transformaciones (La clave para soportar fotos de celular)
# Train: Le metemos ruido, rotación y cambios de color para que sea robusta
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Simula mala luz de celular
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Val: Solo redimensionamos (Evaluamos en limpio)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = SkinCancerDataset(train_df, transform=train_transforms)
val_dataset = SkinCancerDataset(val_df, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- CREACIÓN DEL MODELO TRANSFORMER (ViT) ---
print("\n🧠 Descargando cerebro Vision Transformer (Google ViT)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Usando dispositivo: {device}")

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=NUM_CLASSES
)
model.to(device)

# Optimizador y Función de Pérdida
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# --- BUCLE DE ENTRENAMIENTO ---
print("\n🔥 INICIANDO ENTRENAMIENTO...")

best_accuracy = 0.0

for epoch in range(EPOCHS):
    # --- FASE DE ENTRENAMIENTO ---
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS} [Train]")
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # --- FASE DE VALIDACIÓN ---
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    accuracy = 100 * val_correct / val_total
    print(f"   ✅ Precisión Validación: {accuracy:.2f}%")
    
    # --- GUARDAR SI MEJORA ("NO OLVIDAR LO APRENDIDO") ---
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print(f"   💾 Nuevo récord. Guardando modelo en {os.path.basename(MODEL_SAVE_PATH)}...")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("\n🏆 ENTRENAMIENTO FINALIZADO.")
print(f"Mejor precisión lograda: {best_accuracy:.2f}%")