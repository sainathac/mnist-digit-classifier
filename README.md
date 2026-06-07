# MNIST Digit Classifier

A TensorFlow/Keras dense neural network for handwritten digit classification using the MNIST dataset.

## Overview

This project trains a fully connected neural network to classify 28x28 grayscale images of handwritten digits from the MNIST dataset.

## What This Lab Demonstrates

- Loading the MNIST dataset with Keras
- Pixel normalization
- One-hot encoding labels
- Building a dense neural network with the Keras Sequential API
- Training with validation split
- Evaluating test accuracy and loss
- Plotting training and validation curves
- Running prediction on a single test image

## Model Architecture

```text
Flatten(28 x 28)
Dense(128, ReLU)
Dense(10, Softmax)
```

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the lab:

```bash
python src/main.py
```

## Dataset

The MNIST dataset is loaded directly from `tensorflow.keras.datasets`. No dataset files are committed to this repository.
