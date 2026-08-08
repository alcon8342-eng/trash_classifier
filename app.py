# app.py
import os
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import json

app = Flask(__servidor__)

# ============= CARGAR MODELO =============
print("Cargando modelo...")
model = keras.models.load_model('modelo_trash.h5')
print("✓ Modelo cargado")

# ============= CARGAR CLASES =============
with open('classes.json', 'r') as f:
    classes = json.load(f)
    print(f"✓ Clases: {classes}")

# ============= RUTA PRINCIPAL =============
@app.route('/', methods=['GET'])
def home():
    """Verifica que la API esté funcionando"""
    return jsonify({
        'status': 'OK',
        'mensaje': 'API Trash Classifier lista',
        'categories': classes
    })

# ============= RUTA DE PREDICCIÓN =============
@app.route('/predict', methods=['POST'])
def predict():
    """
    Recibe foto desde ESP32
    Retorna categoría predicha + confianza
    
    Uso: 
    POST /predict
    Body: form-data con campo 'file' = imagen JPEG
    """
    try:
        # Verificar que llegó archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # ============= PROCESAR IMAGEN =============
        print(f"📸 Procesando: {file.filename}")
        
        # Leer imagen desde archivo
        image = Image.open(io.BytesIO(file.read()))
        
        # Convertir a RGB (por si viene en modo diferente)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionar a 224x224 (tamaño que espera el modelo)
        image = image.resize((224, 224))
        
        # Convertir a array numpy
        image_array = np.array(image, dtype=np.float32)
        
        # Normalizar (dividir entre 255)
        image_array = image_array / 255.0
        
        # Agregar dimensión batch (modelo espera [batch, height, width, channels])
        image_array = np.expand_dims(image_array, axis=0)
        
        # ============= PREDICCIÓN =============
        print("🧠 Analizando...")
        predictions = model.predict(image_array, verbose=0)
        
        # Obtener clase predicha (índice máximo)
        predicted_index = np.argmax(predictions[0])
        predicted_class = classes[predicted_index]
        
        # Obtener confianza (probabilidad)
        confidence = float(predictions[0][predicted_index]) * 100
        
        # ============= RESPUESTA =============
        print(f"✓ Resultado: {predicted_class} ({confidence:.1f}%)")
        
        response = {
            'category': predicted_class,
            'confidence': f'{confidence:.1f}%',
            'all_predictions': {
                classes[i]: f'{float(predictions[0][i]) * 100:.1f}%' 
                for i in range(len(classes))
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============= EJECUTAR LOCALMENTE =============
if __servidor__ == '__main__':
    # Para desarrollo local
    app.run(debug=True, host='0.0.0.0', port=5000)