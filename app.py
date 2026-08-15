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
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Respuesta de prueba
        return jsonify({
            'category': 'lata',
            'confidence': '92.5%'
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
