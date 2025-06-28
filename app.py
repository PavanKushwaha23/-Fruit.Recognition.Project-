from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load MobileNet model pretrained on ImageNet (as example)
model = MobileNet(weights='imagenet')

def prepare_image(img, target_size=(224, 224)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        processed_img = prepare_image(img)
        preds = model.predict(processed_img)
        decoded = decode_predictions(preds, top=5)[0]

        # Filter results to fruits only - simple keyword filter
        fruit_keywords = ['fruit', 'apple', 'banana', 'orange', 'lemon', 'strawberry', 
                          'pineapple', 'grape', 'watermelon', 'pear', 'mango', 'pomegranate', 
                          'fig', 'citrus', 'coconut', 'kiwi']

        fruit_preds = [ {'name': item[1], 'probability': float(item[2])} 
                        for item in decoded if any(kw in item[1].lower() for kw in fruit_keywords) ]

        if not fruit_preds:
            # Return top 3 predictions regardless
            fruit_preds = [ {'name': item[1], 'probability': float(item[2])} for item in decoded[:3]]

        return jsonify({'predictions': fruit_preds})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

