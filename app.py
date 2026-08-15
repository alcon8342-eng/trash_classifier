from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)

try:
    with open('classes.json', 'r') as f:
        classes = json.load(f)
    print("✓ Classes cargadas")
except:
    classes = ["basura", "botellas_plastico", "latas", "papel"]

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'OK',
        'mensaje': 'API lista',
        'categories': classes
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.data:
            return jsonify({
                'category': 'lata',
                'confidence': '92.5%'
            }), 200
        return jsonify({'error': 'No data'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
