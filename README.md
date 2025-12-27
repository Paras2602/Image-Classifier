# Image Classifier (CIFAR-10) — CNN + ResNet18 + Streamlit App

This project trains an image classification model on **CIFAR-10** and provides a **Streamlit web app** to upload an image and view **Top‑K predictions with confidence**.

## Dataset
- CIFAR-10 (10 classes): airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  
Source: https://www.cs.toronto.edu/~kriz/cifar.html

## Models
1. **Baseline**: Custom CNN (trained from scratch)
2. **Improved**: **ResNet18 (transfer learning)** with a fine-tuned classification head

## Project Structure

.
├── training.ipynb # training notebook
├── app.py # Streamlit app (Top-K predictions)
├── requirements.txt # runtime dependencies
├── requirements-dev.txt # notebook/dev dependencies
├── models/
│ ├── baseline_best.pt
│ └── cifar10_resnet18_best.pt
└── sample_images/ # sample test images


## Setup
Create and activate a Python environment (recommended), then install dependencies:

```bash
pip install -r requirements-dev.txt

