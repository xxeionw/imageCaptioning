import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image

# 모델 및 Tokenizer 로드
model = tf.keras.models.load_model('captioning_model5000')
with open('tokenizer5000.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# ResNet50 특징 추출 모델 정의
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=global_avg_pooling)

# 캡션 생성 함수
def generate_caption(image_feature, max_length=48):  # 저장된 모델의 max_length와 일치시킴
    caption = [tokenizer.word_index['<start>']]
    for _ in range(max_length):
        padded_caption = tf.keras.utils.pad_sequences([caption], maxlen=max_length, padding='post')
        pred = model.predict([image_feature, padded_caption])
        next_word = np.argmax(pred[0, len(caption) - 1, :])
        if next_word == tokenizer.word_index['<end>']:
            break
        caption.append(next_word)
    return ' '.join([tokenizer.index_word[idx] for idx in caption if idx not in [0, tokenizer.word_index['<start>']]])

# 테스트 이미지 전처리 함수
def preprocess_test_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = preprocess_input(image.img_to_array(img))
    img_array = np.expand_dims(img_array, axis=0)
    return feature_extractor.predict(img_array).flatten()

# 테스트 이미지와 캡션 출력
def display_test_image_with_caption(img_path):
    image_feature = np.expand_dims(preprocess_test_image(img_path), axis=0)
    assert image_feature.shape == (1, 2048), f"Expected image_feature shape (1, 2048), got {image_feature.shape}"

    generated_caption = generate_caption(image_feature, max_length=48)
    img = Image.open(img_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Generated Caption: {generated_caption}", fontsize=12)
    plt.show()

# 테스트 실행
test_image_path = 'testimg.jpg'
display_test_image_with_caption(test_image_path)
