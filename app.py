#from typing import io

from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import os
import json

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTADOR_PATH = os.path.join(BASE_DIR, 'model', 'contador.json')

# Cargar los modelos
model = joblib.load(os.path.join(BASE_DIR, 'model', 'modelo_brasil_solar.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'model', 'modelo_brasil_scaler.pkl'))
pca = joblib.load(os.path.join(BASE_DIR, 'model', 'modelo_brasil_pca.pkl'))

def actualizar_contador():
    if not(os.path.exists(CONTADOR_PATH)):
        with open(CONTADOR_PATH, 'w') as f:
            json.dump({'conteo':0}, f)

    with open(CONTADOR_PATH, 'r') as f:
        data = json.load(f)

    data['conteo'] += 1

    with open(CONTADOR_PATH, 'w') as f:
        json.dump(data, f)

app = Flask(__name__)
@app.route('/')
def hello_world():  # put application's code here
    return render_template('home.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route ('/modelo', methods=['GET', 'POST'])
def modelo():
    return render_template('modelo.html')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        #Obtenemos los datos de entrada en formato JSON
        data = request.get_json(force=True)
        input_data = np.array(data['features']).reshape(1, -1) #una lista numerica

        #Preprocesamiento
        input_scaled = scaler.transform(input_data)
        input_pca = pca.transform(input_scaled)

        #Predicción
        prediction = model.predict(input_pca)
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

from flask import send_file
import io

@app.route('/generar_reporte', methods=['POST'])
def generate_report():
    try:
        data = request.get_json(force=True)
        input_data = np.array(data['features']).reshape(1, -1)

        input_scaled = scaler.transform(input_data)
        input_pca = pca.transform(input_scaled)
        prediction = model.predict(input_pca)[0]

        #sumar contador
        conteo = actualizar_contador()

        # Crear PDF en memoria
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2.0, height - 50, "Reporte de Predicción de Producción de Energía Solar")

        c.setFont("Helvetica", 12)
        c.drawString(50, height - 100, f"Predicción de producción: {prediction:.2f}")
        c.drawString(50, height - 130, "Ingreso de características:")

        x_offset = 50
        y_offset = height - 160
        row_height = 15
        col_width = 100

        for i, feature in enumerate(data['features']):
            x = x_offset + (i % 4) * col_width
            y = y_offset - (i // 4) * row_height
            c.drawString(x, y, str(feature))

        c.save()
        buffer.seek(0)

        return send_file(buffer, as_attachment=True, download_name="reporte.pdf", mimetype='application/pdf')

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/conteo_predicciones', methods=['GET'])
def conteo_predicciones():
    if not os.path.exists(CONTADOR_PATH):
        return jsonify({'conteo': 0})

    with open(CONTADOR_PATH, 'r') as f:
        data = json.load(f)

    return jsonify({'conteo': data.get('conteo', 0)})


if __name__ == '__main__':
    app.run(debug=True)
