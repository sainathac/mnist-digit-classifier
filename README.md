# 🔢 MNIST Digit Classifier

> A clean TensorFlow/Keras dense neural network that achieves high accuracy classifying handwritten digits from the MNIST dataset — with full training visualization and single-image prediction.

---

## 📌 Project Overview

This project trains a fully connected (dense) neural network on the classic MNIST dataset of 70,000 handwritten digit images. It covers the full supervised learning workflow: data loading, normalization, model building, training with validation monitoring, evaluation, and prediction.

A great first real-world deep learning implementation after understanding raw gradient descent.

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Dataset | MNIST (60K train / 10K test) |
| Model Type | Dense Neural Network (MLP) |
| Input Shape | 28×28 grayscale → flattened |
| Output Classes | 10 (digits 0–9) |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |

---

## 🔍 What This Lab Demonstrates

| Concept | Implementation |
|---|---|
| Dataset loading | `tf.keras.datasets.mnist` |
| Preprocessing | Pixel normalization (÷255), one-hot encoding |
| Model architecture | Flatten → Dense(128, ReLU) → Dense(10, Softmax) |
| Training | Validation split, accuracy tracking |
| Evaluation | Test set loss and accuracy |
| Visualization | Training & validation curve plots |
| Prediction | Single test-image inference |

---

## 🏗️ Model Architecture

```text
Flatten(28 × 28)
    ↓
Dense(128, ReLU)
    ↓
Dense(10, Softmax)
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| TensorFlow / Keras | Model building and training |
| NumPy | Data manipulation |
| Matplotlib | Training curve visualization |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/sainathac/mnist-digit-classifier.git
cd mnist-digit-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the lab
```bash
python src/main.py
```

---

## 📚 Part of the Deep Learning Lab Series

This is **Lab 2 of 6** in a TensorFlow/Keras deep learning learning series:

1. [Neural Network From Scratch](https://github.com/sainathac/neural-network-from-scratch)
2. **[MNIST Digit Classifier](https://github.com/sainathac/mnist-digit-classifier)** ← You are here
3. [Fashion-MNIST Model Lifecycle](https://github.com/sainathac/fashion-mnist-model-lifecycle)
4. [Cats vs Dogs CNN Classifier](https://github.com/sainathac/cats-vs-dogs-cnn-classifier)
5. [VGG16 Transfer Learning Classifier](https://github.com/sainathac/vgg16-transfer-learning-classifier)
6. [IMDB Sentiment RNN](https://github.com/sainathac/imdb-sentiment-rnn)

---

## 👤 Author

**Sainatha C**
AI Automation & RPA Engineer | Data Science Practitioner

[![GitHub](https://img.shields.io/badge/GitHub-sainathac-181717?logo=github)](https://github.com/sainathac)

