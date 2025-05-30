from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar los modelos
model = joblib.load(os.path.join(BASE_DIR, 'model', 'modelo_brasil_solar.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'model', 'modelo_brasil_scaler.pkl'))
pca = joblib.load(os.path.join(BASE_DIR, 'model', 'modelo_brasil_pca.pkl'))

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

if __name__ == '__main__':
    app.run(debug=True)
