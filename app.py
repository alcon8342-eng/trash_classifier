from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)

# Cargar clases
with open('classes.json', 'r') as f:
    classes = json.load(f)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'OK',
        'mensaje': 'API Trash Classifier lista',
        'categories': classes
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Opción 1: Si viene como archivo multipart (desde navegador/cliente)
        if 'file' in request.files:
            file = request.files['file']
            file_data = file.read()
        
        # Opción 2: Si viene como datos binarios directos (desde ESP32)
        elif request.data:
            file_data = request.data
        
        else:
            return jsonify({'error': 'No file or data provided'}), 400
        
        # Verificar que recibimos datos JPEG válidos
        if not file_data or len(file_data) < 100:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Por ahora: respuesta de prueba (sin ML)
        # TODO: Aquí va la lógica de clasificación con TensorFlow
        
        return jsonify({
            'category': 'lata',
            'confidence': '92.5%'
        }), 200
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
