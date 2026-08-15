from flask import Flask, request, jsonify
import json
import os
import io
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Cargar clases
try:
    with open('classes.json', 'r') as f:
        classes = json.load(f)
    print("✓ Classes cargadas correctamente")
except Exception as e:
    print(f"✗ Error cargando classes.json: {e}")
    classes = ["basura", "botellas_plastico", "latas", "papel"]

# Cargar modelo TFLite
try:
    interpreter = tflite.Interpreter(model_path='modelo_trash.tflite')
    interpreter.allocate_tensors()
    print("✓ Modelo TFLite cargado correctamente")
except Exception as e:
    print(f"✗ Error cargando modelo: {e}")
    interpreter = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'OK',
        'mensaje': 'API Trash Classifier lista',
        'categories': classes,
        'modelo_cargado': interpreter is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recibir imagen
        if 'file' in request.files:
            file = request.files['file']
            file_data = file.read()
        elif request.data:
            file_data = request.data
        else:
            return jsonify({'error': 'No file or data provided'}), 400
        
        # Convertir a imagen
        imagen = Image.open(io.BytesIO(file_data))
        
        # Redimensionar a 224x224
        imagen = imagen.resize((224, 224))
        
        # Convertir a array
        imagen_array = np.array(imagen, dtype=np.float32) / 255.0
        
        # Ejecutar predicción
        if interpreter is None:
            return jsonify({'error': 'Modelo no cargado'}), 500
        
        # Obtener detalles del modelo
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Ajustar entrada
        interpreter.set_tensor(input_details[0]['index'], [imagen_array])
        interpreter.invoke()
        
        # Obtener resultado
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediccion = output_data[0]
        
        # Clase con mayor probabilidad
        clase_idx = np.argmax(prediccion)
        confianza = float(prediccion[clase_idx]) * 100
        
        # Nombre de la clase
        if isinstance(classes, dict):
            clase_nombre = list(classes.keys())[clase_idx]
        else:
            clase_nombre = classes[clase_idx]
        
        print(f"✓ Predicción: {clase_nombre} ({confianza:.2f}%)")
        
        return jsonify({
            'category': clase_nombre,
            'confidence': f'{confianza:.2f}%'
        }), 200
    
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
