from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# TensorFlow 모델 로드
model = tf.keras.models.load_model('C:/Users/UserK/Desktop/model.keras')

# Nginx 서버에서 이미지를 가져오는 함수
def fetch_image_from_nginx(url):
    try:
        response = requests.get(url)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error fetching image: {e}")
        return None

# 이미지 예측 함수 정의
def predict_image(model, img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # 이미지 정규화
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가
    
    prediction = model.predict(img)
    
    if prediction[0] > 0.5:
        return 'realtime_image.jpg 수소 취화된 이미지'
    else:
        return 'realtime_image.jpg 수소 취화되지 않은 이미지'

# 기본 페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html')

# 이미지 예측 API
@app.route('/predict', methods=['GET'])
def predict():
    image_url = "http://192.168.1.116/images/realtime_image.jpg"
    img = fetch_image_from_nginx(image_url) 

    if img is None:
        return jsonify({'error': 'Failed to fetch image from URL'}), 400

    result = predict_image(model, img)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

