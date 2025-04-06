import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 이미지 파일 경로 설정
image_dir = 'C:/Users/UserK/test_real'

# 데이터 전처리 함수
def load_and_preprocess_images(image_dir, img_size=(224, 224)):
    images = []
    labels = []
    
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = img / 255.0  # 이미지 정규화
        images.append(img)
        # 예시: 파일명에 따라 레이블을 지정할 수 있습니다.
        label = 1 if 'hydrogen_embrittlement' in img_file else 0
        labels.append(label)
    
    return np.array(images), np.array(labels)

# 이미지 및 레이블 로드
X, y = load_and_preprocess_images(image_dir)

# CNN 모델 구성 함수
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 이진 분류를 위한 시그모이드 활성화 함수
    ])
    return model

# 모델 생성 및 컴파일
model = create_model(X.shape[1:])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X, y, epochs=1, batch_size=1, validation_split=0.2)

