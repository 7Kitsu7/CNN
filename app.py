from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from utils import preprocess_for_model, dice_coefficient
import traceback
import google.generativeai as genai  # Añade al inicio del archivo
import requests  # Añade esto al inicio del archivo
import datetime  # <-- Añade esta línea

app = Flask(__name__)
CORS(app)  # Permitir CORS para Flutter

# Cargar modelos una vez al iniciar
models = {
    'pneumonia': load_model('model/best_pneumonia.keras'),
    'lung': load_model('model/best_pulmonar.keras', custom_objects={'dice_coefficient': dice_coefficient}),
    'damage': load_model('model/best_daño.keras')
}
genai.configure(api_key='AIzaSyALYl3jq_eF-Wr269SA6Ffm_CsVhBogOeA')  # Reemplaza con tu API key

IMG_SIZES = {
    'clf_size': (224, 224),
    'seg_size': (512, 512),
    'damage_size': (128, 128)
}

def decode_base64_image(img_base64):
    img_bytes = base64.b64decode(img_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')  # Grayscale
    img_np = np.array(img)
    return img_np

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    img_base64 = data.get('imagen')
    edad = data.get('edad')
    sexo = data.get('sexo')

    if img_base64 is None or edad is None or sexo is None:
        return jsonify({'error': 'Faltan datos'}), 400

    try:
        # Decodificar imagen base64 a numpy array grayscale
        original_img = decode_base64_image(img_base64)

        # Preprocesar imagen para clasificación, segmentación y daño
        clf_img = preprocess_for_model(original_img, IMG_SIZES['clf_size'])
        seg_img = preprocess_for_model(original_img, IMG_SIZES['seg_size'])
        damage_img = preprocess_for_model(original_img, IMG_SIZES['damage_size'])

        # Expandir dimensiones para batch
        clf_input = np.expand_dims(clf_img, axis=0)  # (1, 224, 224, 1)
        seg_input = np.expand_dims(seg_img, axis=0)  # (1, 512, 512, 1)
        damage_input = np.expand_dims(damage_img, axis=0)  # (1, 128, 128, 1)

        # Conversión segura de edad y sexo
        try:
            edad = float(edad)  # aseguramos tipo numérico
            sexo = sexo.lower().strip()
            if sexo == 'masculino':
                sexo_bin = 1
            elif sexo == 'femenino':
                sexo_bin = 0
            else:
                return jsonify({'error': 'Sexo inválido: usa Masculino o Femenino'}), 400
        except Exception as e:
            return jsonify({'error': f'Dato inválido: {str(e)}'}), 400

        # Tabular input para edad y sexo (ambos como float)
        tabular_data = np.array([[edad, sexo_bin]], dtype=np.float32)


        # Predicción neumonía
        pneumonia_prob = models['pneumonia'].predict([clf_input, tabular_data])[0][0]
        diagnosis = "NEUMONÍA" if pneumonia_prob > 0.5 else "Normal"

        # Segmentación pulmonar
        lung_pred = models['lung'].predict(seg_input)[0, ..., 0]
        lung_mask = (lung_pred > 0.5).astype(np.uint8)

        # Detección de daño pulmonar
        pred_damage = models['damage'].predict(damage_input)[0]
        pred_mask = (pred_damage > 0.15).astype(np.uint8)
        damage_mask = cv2.resize(pred_mask.squeeze(), IMG_SIZES['seg_size'], interpolation=cv2.INTER_NEAREST)

        # Cálculo porcentaje daño pulmonar
        damage_perc = (np.sum(lung_mask * damage_mask) / np.sum(lung_mask)) * 100 if np.sum(lung_mask) > 0 else 0.0

        # Retornar resultados
        return jsonify({
            'diagnostico': diagnosis,
            'probabilidad': float(pneumonia_prob),
            'daño_pulmonar': float(damage_perc)
        })

    except Exception as e:
        print("ERROR EN PREDICCIÓN:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    data = request.get_json()
    
    # Datos del diagnóstico recibidos desde Flutter
    diagnostico = data.get('diagnostico')
    probabilidad = data.get('probabilidad')
    dano_pulmonar = data.get('dano_pulmonar')
    edad = data.get('edad')
    sexo = data.get('sexo')

    # Validación básica
    if None in [diagnostico, probabilidad, dano_pulmonar, edad, sexo]:
        return jsonify({'error': 'Faltan datos para la recomendación'}), 400

    try:
        # Modelo Gemini (usa 'gemini-pro' o 'gemini-2.0-flash')
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Prompt estructurado para Gemini
        prompt = f"""
        Eres un médico experto en neumología. Genera una recomendación médica basada en estos datos que se obtuvieron de la radiografia de torax y datos del paciente:

        - **Edad**: {edad} años
        - **Sexo**: {sexo}
        - **Diagnóstico preliminar**: {diagnostico}
        - **Probabilidad de neumonía**: {probabilidad * 100:.1f}%
        - **Daño pulmonar estimado**: {int(dano_pulmonar)}%

        Debe incluir lo siguiente :
        1. Explicación clara del diagnóstico en términos no técnicos.
        2. Acciones inmediatas (ej: medicamentos, reposo).
        3. Recomendaciones de seguimiento (ej: especialista, pruebas).
        4. Precauciones generales (ej: evitar fumar).

        Responde en español. Tu respuesta debe ser corta y precisa.
        """

        # Consulta a Gemini
        response = model.generate_content(prompt)
        recomendacion = response.text

        # 2. Preparar datos para el webhook
        webhook_data = {
            "paciente": {
                "edad": edad,
                "sexo": sexo
            },
            "diagnostico": {
                "resultado": diagnostico,
                "probabilidad_neumonia": f"{probabilidad * 100:.1f}%",
                "dano_pulmonar": f"{int(dano_pulmonar)}%"
            },
            "recomendacion": recomendacion,
            "timestamp": datetime.datetime.now().isoformat()
        }

        # 3. Enviar al webhook
        webhook_url = "https://wazilrest-n8n.xwk85y.easypanel.host/webhook/1733b542-3069-4623-8d55-614b9e3a6172"
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        webhook_response = requests.post(
            webhook_url,
            json=webhook_data,
            headers=headers,
            timeout=10  # 10 segundos de timeout
        )

        # Verificar respuesta del webhook
        if webhook_response.status_code not in [200, 201]:
            return jsonify({
                'error': f'Error al enviar al webhook: {webhook_response.text}',
                'recomendacion': recomendacion
            }), 500
        
        return jsonify({
            'recomendacion': recomendacion
        })

    except Exception as e:
        return jsonify({'error': f'Error al consultar Gemini: {str(e)}'}), 500
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
