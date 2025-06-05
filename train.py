import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention
from tensorflow.keras.models import Model
import numpy as np
import os
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import random

# 1. ResNet50 기반 특징 추출 모델 정의
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=global_avg_pooling)

# 2. COCO 데이터셋 로드
train_images_dir = r'C:\Users\DS\Desktop\딥러닝응용\val2014'
annotations_file = r'C:\Users\DS\Desktop\딥러닝응용\annotations\captions_val2014.json'
coco = COCO(annotations_file)
image_ids = coco.getImgIds()[:5000]

# 3. 이미지 전처리 함수
def preprocess_image(img_id):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(train_images_dir, img_info['file_name'])
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = preprocess_input(image.img_to_array(img))
    img_array = np.expand_dims(img_array, axis=0)
    return feature_extractor.predict(img_array).flatten()

# 4. 캡션 데이터 전처리
all_captions = []
image_features = {}
for img_id in tqdm(image_ids, desc="Processing Images"):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    all_captions.extend([ann['caption'] for ann in anns])
    image_features[img_id] = preprocess_image(img_id)

# 토큰화 및 단어 인덱스 매핑
tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(all_captions)
tokenizer.word_index['<pad>'] = 0
tokenizer.word_index['<start>'] = len(tokenizer.word_index) + 1
tokenizer.word_index['<end>'] = len(tokenizer.word_index) + 1

# 캡션 인코딩 및 패딩
def preprocess_captions(captions):
    seqs = tokenizer.texts_to_sequences(captions)
    return [[tokenizer.word_index['<start>']] + seq + [tokenizer.word_index['<end>']] for seq in seqs]

encoded_captions = {
    img_id: preprocess_captions([ann['caption'] for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id))])
    for img_id in image_ids
}
max_caption_length = max(len(seq) for captions in encoded_captions.values() for seq in captions)
padded_captions = {
    img_id: pad_sequences(captions, maxlen=max_caption_length, padding='post', value=0)
    for img_id, captions in encoded_captions.items()
}

# 5. 훈련 데이터 준비
image_features_array = []
captions_array = []
for img_id, features in image_features.items():
    for caption in padded_captions[img_id]:
        image_features_array.append(features)
        captions_array.append(caption)
image_features_array = np.array(image_features_array)
captions_array = np.array(captions_array)

# 6. 데이터셋 분할
train_image_features, test_image_features, train_captions, test_captions = train_test_split(
    image_features_array, captions_array, test_size=0.2, random_state=42
)

# 7. 캡션 생성 모델 정의 (Encoder-Decoder 구조)
class AttentionCaptioningModel(tf.keras.Model):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        super().__init__()
        self.embedding = Embedding(vocab_size, wordvec_size)
        self.lstm = LSTM(hidden_size, return_sequences=True, return_state=True)
        self.attention = Attention()
        self.dense_image_features = Dense(hidden_size, activation='relu')
        self.output_layer = Dense(vocab_size, activation='softmax')

    def call(self, inputs, training=False):
        image_features, captions = inputs
        image_features = tf.expand_dims(self.dense_image_features(image_features), axis=1)
        embedded_captions = self.embedding(captions)
        lstm_out, _, _ = self.lstm(embedded_captions, initial_state=[image_features[:, 0, :], image_features[:, 0, :]])
        context_vector = self.attention([lstm_out, image_features])
        combined_features = tf.concat([context_vector, lstm_out], axis=-1)
        return self.output_layer(combined_features)

# 8. 모델 컴파일 및 훈련
vocab_size = len(tokenizer.word_index) + 1
wordvec_size = 128
hidden_size = 512
model = AttentionCaptioningModel(vocab_size, wordvec_size, hidden_size)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    [train_image_features, train_captions[:, :-1]],
    train_captions[:, 1:],
    epochs=50,
    batch_size=128
)

# 9. 캡션 생성 함수
def generate_caption(image_feature, max_length=20):
    caption = [tokenizer.word_index['<start>']]
    for _ in range(max_length):
        pred = model.predict([image_feature, np.array([caption])])
        next_word = np.argmax(pred[0, -1, :])
        if next_word == tokenizer.word_index['<end>']:
            break
        caption.append(next_word)
    return ' '.join([tokenizer.index_word[idx] for idx in caption if idx not in [0, tokenizer.word_index['<start>']]])

# 10. BLEU 점수 계산 함수
# BLEU 점수 계산 함수 (1-gram, 2-gram, 3-gram, 4-gram)
def evaluate_bleu(test_features, test_input, test_output):
    predictions = []
    references = []
    for feature, input_seq, output_seq in zip(test_features, test_input, test_output):
        feature = np.expand_dims(feature, axis=0)  # 이미지를 (1, feature_size) 형태로 확장
        input_seq = input_seq.reshape(1, -1)  # 캡션 시퀀스도 (1, seq_length) 형태로 확장
        prediction = model.predict([feature, input_seq])  # 예측 수행
        predicted_seq = np.argmax(prediction, axis=-1).flatten()  # 예측된 시퀀스

        # 예측된 시퀀스 및 실제 시퀀스를 BLEU 계산에 맞게 변환
        predictions.append(predicted_seq)
        references.append([output_seq])

    # BLEU-1, BLEU-2, BLEU-3, BLEU-4 계산
    bleu_1 = corpus_bleu(references, predictions, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(references, predictions, weights=(0, 1.0, 0, 0))
    bleu_3 = corpus_bleu(references, predictions, weights=(0, 0, 1.0, 0))
    bleu_4 = corpus_bleu(references, predictions, weights=(0, 0, 0, 1.0))

    # 평균 BLEU 점수 계산
    bleu_avg = (bleu_1 + bleu_2 + bleu_3 + bleu_4) / 4

    return bleu_1, bleu_2, bleu_3, bleu_4, bleu_avg

# 10. 이미지와 캡션 출력
def display_image_with_caption(image_id, generated_caption):
    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(train_images_dir, img_info['file_name'])
    img = Image.open(img_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Generated Caption: {generated_caption}", fontsize=12)
    plt.show()

def display_random_image_with_caption():
    random_image_id = random.choice(image_ids)
    random_image_feature = np.array([image_features[random_image_id]])
    print('generated_caption 함수')
    generated_caption = generate_caption(random_image_feature)
    print('display_image_with_caption 함수')
    display_image_with_caption(random_image_id, generated_caption)

def calculate_accuracy(predicted, actual):
    """
    단어의 공통 개수를 기반으로 정확도를 계산하는 함수.
    Args:
        predicted: 모델이 예측한 캡션의 인덱스 배열.
        actual: 실제 캡션의 인덱스 배열.

    Returns:
        accuracy: 예측된 캡션과 실제 캡션 간의 공통 단어 비율로 계산된 정확도.
    """
    correct = 0
    total = 0

    for pred, true in zip(predicted, actual):
        # 예측된 캡션의 단어 리스트
        pred_caption = [tokenizer.index_word[idx] for idx in pred if
                        idx not in [0, tokenizer.word_index['<start>'], tokenizer.word_index['<end>']]]

        # 실제 캡션의 단어 리스트
        true_caption = [tokenizer.index_word[idx] for idx in true if
                        idx not in [0, tokenizer.word_index['<start>'], tokenizer.word_index['<end>']]]

        # 공통 단어 개수 비교
        common_words = set(pred_caption) & set(true_caption)
        correct += len(common_words)
        total += len(true_caption)

    # 정확도 계산
    accuracy = correct / total if total > 0 else 0
    return accuracy


# 예측된 캡션 생성
predictions = []
for img_feature in test_image_features:
    predicted_caption = generate_caption(np.array([img_feature]))
    predictions.append([tokenizer.word_index[word] for word in predicted_caption.split() if word in tokenizer.word_index])

#평가 함수 실행
#bleu_1, bleu_2, bleu_3, bleu_4, bleu_avg = evaluate_bleu(test_image_features, test_captions[:, :-1], test_captions[:, 1:])
#print(f"BLEU-1: {bleu_1}")
#print(f"BLEU-2: {bleu_2}")
#print(f"BLEU-3: {bleu_3}")
#print(f"BLEU-4: {bleu_4}")
#print(f"Average BLEU: {bleu_avg}")


# 단어 공통 개수 기반 테스트 정확도 계산
test_accuracy = calculate_accuracy(predictions, test_captions[:, 1:])
print(f"Test accuracy: {test_accuracy:.4f}")

# 랜덤 이미지와 캡션 생성 및 출력
display_random_image_with_caption()

# 모델 저장 (SavedModel 형식)
model.save('captioning_model5000', save_format='tf')

# Tokenizer 저장
import pickle
with open('tokenizer5000.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved successfully.")