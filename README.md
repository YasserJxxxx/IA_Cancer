# 🧬 DermaAI: Diagnóstico Dermatológico Inteligente

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c)
![Transformers](https://img.shields.io/badge/HuggingFace-ViT-yellow)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![License](https://img.shields.io/badge/License-MIT-green)

> **Sistema de apoyo al pre-diagnóstico de cáncer de piel utilizando Vision Transformers (ViT) y Aprendizaje Continuo.**

---

## 📋 Tabla de Contenidos
- [Sobre el Proyecto](#-sobre-el-proyecto)
- [Características Principales](#-características-principales)
- [Arquitectura y Datos](#-arquitectura-y-datos)
- [Demo](#-demo)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Resultados y Métricas](#-resultados-y-métricas)
- [Descargo de Responsabilidad](#-descargo-de-responsabilidad)

---

## 📖 Sobre el Proyecto

**DermaAI** nace de la necesidad de democratizar el acceso al triaje dermatológico temprano. El melanoma es altamente curable si se detecta a tiempo, pero la falta de acceso a especialistas y herramientas de diagnóstico crea una barrera mortal.

Este proyecto implementa un modelo de **Inteligencia Artificial (Vision Transformer)** entrenado con una estrategia híbrida: combina imágenes médicas de alta calidad (Dermatoscopia) con imágenes tomadas por smartphones, permitiendo que el sistema sea robusto en condiciones reales de uso doméstico.

---

## ✨ Características Principales

* 🔍 **Detección Multiclase:** Clasifica 7 tipos de lesiones cutáneas (Melanoma, Nevus, Carcinomas, etc.).
* 📱 **Soporte Móvil:** Diseñado para funcionar con fotos de celular a través de una Web App responsiva.
* 🧠 **Vision Transformer (ViT):** Utiliza mecanismos de *Self-Attention* para detectar patrones asimétricos sutiles mejor que las CNN tradicionales.
* 🚦 **Sistema de Semáforo:** * 🟢 Benigno (Observación)
    * 🟠 Precaución (Seguimiento)
    * 🔴 Peligro (Atención Inmediata)
* 🛡️ **Control de Calidad:** Si la IA detecta baja confianza (<70%), solicita al usuario mejorar la foto.
* 🔄 **Aprendizaje Continuo:** Guarda automáticamente los nuevos casos (`dataset_nuevos_casos`) para re-entrenar y mejorar el modelo con el tiempo.

---

## 🏗️ Arquitectura y Datos

### El Dataset Híbrido
Para mitigar el sesgo de laboratorio, unificamos dos fuentes de datos:

| Dataset | Tipo | Propósito |
| :--- | :--- | :--- |
| **HAM10000** | Dermatoscopia | Aprender texturas celulares finas. |
| **PAD-UFES-20** | Clínica (Celular) | Aprender a manejar sombras, luz variable y ruido. |

### Stack Tecnológico
* **Modelado:** PyTorch, Hugging Face Transformers.
* **Procesamiento de Datos:** Pandas, NumPy, PIL.
* **Interfaz:** Streamlit (Python puro).
* **Despliegue Remoto:** Ngrok (Túnel seguro para acceso móvil).

---

## 📸 Demo

*(Aquí puedes poner capturas de pantalla de tu App. Reemplaza las rutas de abajo con tus imágenes en la carpeta 'analisis' o capturas de tu celular)*

| Interfaz de Carga | Análisis en Tiempo Real | Resultados |
| :---: | :---: | :---: |
| ![Home](https://via.placeholder.com/200x400?text=App+Home) | ![Scanning](https://via.placeholder.com/200x400?text=Escaner) | ![Result](https://via.placeholder.com/200x400?text=Diagnostico) |

---

## ⚙️ Instalación y Configuración

### Prerrequisitos
* Python 3.8 o superior.
* CUDA (Opcional, recomendado para entrenamiento rápido).
* Cuenta de [Ngrok](https://ngrok.com) (para usar en el celular).

### 1. Clonar el repositorio
```bash
git clone [[https://github.com/TU_USUARIO/DermaAI.git](https://github.com/TU_USUARIO/DermaAI.git)](https://github.com/YasserJxxxx/IA_Cancer.git)
cd DermaAI
2. Crear entorno virtual
Bash

python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
3. Instalar dependencias
Bash

pip install -r requirements.txt
4. Preparar los datos
Coloca los archivos cancer.zip (HAM10000) y cancer2.zip (PAD-UFES) en la carpeta dataset/ y ejecuta:

Bash

python unificar_datasets.py
🚀 Uso
A. Entrenar el Modelo (Opcional si ya tienes el .pth)
Si deseas re-entrenar el cerebro de la IA:

Bash

python entrenar_ia.py
B. Ejecutar la Aplicación (Modo Local)
Para abrir la interfaz visual en tu PC:

Bash

streamlit run app_ui.py
C. Conectar al Celular (Modo Remoto)
Para generar un enlace accesible desde tu smartphone:

Bash

# En una nueva terminal
python conectar_app.py
Copia la URL generada (ej. https://xxxx.ngrok-free.app) y ábrela en tu móvil.

📂 Estructura del Proyecto
Plaintext

DermaAI/
├── dataset/                  # Datos crudos y CSV unificado
├── dataset_nuevos_casos/     # Fotos recolectadas por la App (Active Learning)
├── analisis/                 # Gráficos de evaluación y métricas
├── modelo_cancer_piel_vit.pth # Pesos del modelo entrenado
├── app_ui.py                 # Código de la Interfaz (Frontend)
├── api_cancer.py             # Código de la API (Backend)
├── entrenar_ia.py            # Script de entrenamiento
├── conectar_app.py           # Script de conexión Ngrok
└── requirements.txt          # Librerías necesarias
📊 Resultados y Métricas
El modelo ha sido evaluado con un set de validación del 20% (imágenes nunca vistas).

Sensibilidad (Recall) Melanoma: > 85% (Prioridad Alta)

Accuracy Global: ~88%

(Matriz de confusión generada durante la fase de evaluación)

⚠️ Descargo de Responsabilidad
IMPORTANTE: DermaAI es una herramienta de investigación y apoyo educativo. Los resultados proporcionados son probabilísticos y NO constituyen un diagnóstico médico.

Esta herramienta puede cometer errores.

No sustituye la consulta con un dermatólogo profesional.

Ante cualquier duda o cambio en una lesión, acuda siempre a un médico.

📄 Licencia
Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE.md para más detalles.

Desarrollado con ❤️ por El grupo Specter
