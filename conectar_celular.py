import sys
from pyngrok import ngrok

# Configura el puerto donde corre tu API (api_cancer.py)
PORT = 8000

print("🌍 Creando túnel hacia internet...")

# Abrir el túnel HTTP
try:
    public_url = ngrok.connect(PORT).public_url
    print("\n" + "="*60)
    print(f"📲 ¡ENLACE PARA TU CELULAR GENERADO!")
    print(f"👉 {public_url}/docs")
    print("="*60)
    print("\nInstrucciones:")
    print("1. Copia el enlace de arriba.")
    print("2. Mándatelo por WhatsApp o Telegram.")
    print("3. Ábrelo en tu celular.")
    print("4. Usa la interfaz para tomar una foto con la cámara.")
    print("\n(Presiona Ctrl+C para cerrar el túnel)")

    # Mantiene el script vivo
    try:
        # Loop infinito compatible con Python 3
        while True:
            pass
    except KeyboardInterrupt:
        print("\nCerrando túnel...")
        ngrok.kill()
except Exception as e:
    print(f"❌ Error: {e}")
    print("Nota: Si es la primera vez, quizás necesites registrarte gratis en ngrok.com y poner tu token.")