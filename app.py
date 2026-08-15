import os
import json
import io
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# ============= CARGAR MODELO TFLITE Y CLASES =============
print("Cargando modelo TFLite...")
interpreter = tflite.Interpreter(model_path="modelo_trash.tflite")
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✓ Modelo TFLite cargado correctamente")

with open('classes.json', 'r') as f:
    classes = json.load(f)
    print(f"✓ Clases cargadas: {classes}")

# ============= RUTAS =============
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'OK',
        'mensaje': 'API Trash Classifier TFLite lista',
        'categories': classes
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"📸 Procesando: {file.filename}")
        
        # 1. Leer y procesar la imagen
        image = Image.open(io.BytesIO(file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionar según lo que espera la entrada del modelo
        input_shape = input_details[0]['shape'] # [1, height, width, 3]
        height, width = input_shape[1], input_shape[2]
        image = image.resize((width, height))
        
        # Normalizar a float32 entre 0.0 y 1.0
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # 2. Inferencia con TFLite
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # 3. Formatear respuesta
        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = classes[predicted_index]
        confidence = float(predictions[0][predicted_index]) * 100
        
        return jsonify({
            'category': predicted_class,
            'confidence': f'{confidence:.1f}%',
            'all_predictions': {
                classes[i]: f'{float(predictions[0][i]) * 100:.1f}%' 
                for i in range(len(classes))
            }
        }), 200

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
