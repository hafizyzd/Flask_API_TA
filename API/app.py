from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from utils import load_tokenizer, preprocess_text
from flask_cors import CORS 
import os

app = Flask(__name__)
CORS(app,origins=["http://127.0.0.1:8000"])

model_path = os.path.join(os.path.dirname(__file__), 'cnn_lstm_model.h5')  
tokenizer_path = os.path.join(os.path.dirname(__file__), 'tokenizer.pkl')  

model = load_model(model_path)
tokenizer = load_tokenizer(tokenizer_path)

MAX_SEQUENCE_LENGTH = 100  

@app.route('/detect_sqli', methods=['POST'])
def detect_sqli():
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({
                'error': 'Bad request',
                'message': 'Input JSON with field "query"'
            }), 400
        
        query = data['query']
        
        processed_text = preprocess_text(
            text=query,
            tokenizer=tokenizer,
            max_len=MAX_SEQUENCE_LENGTH
        )
        
        prediction = model.predict(processed_text)
        probability = float(prediction[0][0])
        is_sqli = probability > 0.5  
        
        response = {
            'query': query,
            'is_sqli': is_sqli,
            'probability': probability,
        }
        print(response)
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)