import cv2
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO

# 학습된 모델 로드
model = tf.keras.models.load_model('C:/Users/UserK/Desktop/model.keras')

# Nginx 서버에서 이미지를 가져오는 함수
def fetch_image_from_nginx(url):
    response = requests.get(url)
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

# 이미지 예측 함수 정의
def predict_image(model, img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # 이미지 정규화
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가
    
    prediction = model.predict(img)
    
    if prediction[0] > 0.5:
        return "수소 취화된 이미지"
    else:
        return "수소 취화되지 않은 이미지"

# 예측 예시
nginx_image_url = 'http://192.168.1.116/images/realtime_image.jpg'
img = fetch_image_from_nginx(nginx_image_url)
result = predict_image(model, img)
print(result)