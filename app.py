#from typing import io

from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import os
import json
import io

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch

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

@app.route('/introduccion')
def introduccion():  # put application's code here
    return render_template('acad_introduccion.html')

@app.route('/conclusiones')
def conclusiones():  # put application's code here
    return render_template('acad_conclusiones.html')

@app.route('/bibliografia')
def bibliografia():  # put application's code here
    return render_template('acad_bibliografia.html')

@app.route('/dataset')
def dataset():  # put application's code here
    return render_template('acad_dataset.html')

@app.route('/desarrollo')
def desarrollo():  # put application's code here
    return render_template('acad_desarrollo.html')

@app.route('/marco')
def marco():  # put application's code here
    return render_template('acad_marco.html')

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

@app.route('/generar_reporte', methods=['POST'])
def generate_report():
    try:
        data = request.get_json(force=True)
        input_data = np.array(data['features']).reshape(1, -1)

        input_scaled = scaler.transform(input_data)
        input_pca = pca.transform(input_scaled)
        prediction = model.predict(input_pca)[0]

        # Sumar contador
        conteo = actualizar_contador()

        # Crear PDF en memoria
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Banner con logo y texto
        logo_path = "static/dist/img/utpl_logo.png"
        c.drawImage(logo_path, 50, height-60, width=60, preserveAspectRatio=True)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(120, height - 60, "Sistema de Predicción de Energía Solar")

        # Texto introductorio
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 110, "Este reporte presenta la predicción de producción de energía solar basada en los datos ingresados.")
        c.drawString(50, height - 130, "El modelo ha sido entrenado con técnicas de aprendizaje automático para ofrecer estimaciones precisas.")

        # Resultado de la predicción
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.darkblue)
        c.drawString(50, height - 170, f"Predicción estimada: {prediction:.2f} kWh")
        c.setFillColor(colors.black)

        # Características de entrada
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height - 200, "Características de entrada:")
        c.setFont("Helvetica", 10)

        x_offset = 50
        y_offset = height - 220
        row_height = 15
        col_width = 130

        for i, feature in enumerate(data['features']):
            x = x_offset + (i % 3) * col_width
            y = y_offset - (i // 3) * row_height
            c.drawString(x, y, f"Característica {i+1}: {feature}")

        # Cuadro de resumen al final
        c.setFont("Helvetica-Bold", 10)
        c.rect(50, 50, width - 100, 60, stroke=1, fill=0)
        c.drawString(60, 90, "Aplicación: Predicción Solar")
        c.drawString(60, 75, "Proyecto: Estimación de Producción Fotovoltaica")
        c.drawString(60, 60, "Página: 1")

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

@app.route('/<seccion>')
def mostrar_seccion(seccion):
    try:
        return render_template('home.html', seccion=seccion)
    except:
        return "Sección no encontrada", 404


if __name__ == '__main__':
    app.run()
