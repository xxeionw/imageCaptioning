# 🧠 이미지 캡셔닝 (Image Captioning)

이미지를 입력으로 받아 자연어 설명을 생성하는 딥러닝 기반 프로젝트입니다.  
CNN을 통해 이미지 특징을 추출하고, LSTM 기반의 Seq2Seq 모델로 설명 문장을 생성합니다.

---

## 📌 프로젝트 개요

- **목표**: 이미지에 대한 자연스러운 문장 설명 생성
- **모델 구조**:
  - **CNN (resnet50)**: 이미지 feature 추출
  - **LSTM + Attention**: 단어 시퀀스 생성
- **데이터셋**: MS COCO (소규모 버전 사용)
- **성능 평가**: BLEU Score

---

## 🛠 사용 기술

- Python
- TensorFlow / Keras
- Matplotlib, NumPy

---

## 📷 결과 예시
![result](https://github.com/user-attachments/assets/46044970-bdda-4daa-a0b9-ad85403519ad)
![peeky](https://github.com/user-attachments/assets/0e654a3f-6892-47d2-9981-7dc12227c88a)
![attention](https://github.com/user-attachments/assets/6f159c2c-865e-4fbf-855a-ae17a2160981)

---

## 🧠 배운 점

- 딥러닝 모델 설계 및 학습 파이프라인 구성
- CNN-LSTM 연동 구조 이해
- Attention 메커니즘의 역할과 시각화

