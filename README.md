# 🌾 Early Detection of Wheat Disease

An AI-powered deep learning system for the early detection and classification of wheat leaf diseases using computer vision and image processing.

<p align="center">

**🌱 AI-Powered Agriculture • 🔍 Early Disease Detection • 🤖 Deep Learning • 🌾 Smart Farming**

</p>

---

## 📌 Overview

**Early Detection of Wheat Disease** is an AI-based agricultural application designed to identify common wheat leaf diseases from images.

The system uses **Deep Learning, Convolutional Neural Networks (CNN), and Transfer Learning** to classify wheat leaves into different disease categories.

The project provides a simple web interface where users can upload a wheat leaf image and receive:

- 🌾 Disease classification
- 📊 Prediction confidence
- 🔍 Disease information
- ⚠️ Severity information
- 💊 Management recommendations
- 🛡️ Prevention strategies

The goal is to support **early identification of wheat diseases**, helping reduce crop losses and improve agricultural decision-making.

---

# ✨ Features

### 🌱 Automatic Disease Classification

The system can classify wheat leaves into:

- 🟢 Healthy
- 🔴 Leaf Rust
- 🟡 Powdery Mildew
- 🔴 Septoria Leaf Blotch

---

### 🤖 Deep Learning Models

The project supports multiple model architectures:

- **MobileNetV2**
  - Fast and lightweight
  - Transfer learning
  - Suitable for deployment

- **Custom CNN**
  - Lightweight architecture
  - Suitable for research and experimentation
  - Smaller model size

---

### 🖥️ Web Application

The Flask-based web application provides:

- 📤 Image upload
- 🖱️ Drag-and-drop support
- ⚡ Real-time prediction
- 📊 Confidence score
- 📋 Disease information
- 💡 Management recommendations
- 📈 Prediction statistics

---

### 💻 Command-Line Tools

The project also provides command-line support for:

- Model training
- Single-image prediction
- Batch prediction
- Dataset processing
- Result export
- Visualization

---

# 🏗️ System Architecture

```text
                ┌─────────────────────┐
                │   Wheat Leaf Image  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Image Preprocessing │
                │ Resize / Normalize  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Deep Learning     │
                │      Model          │
                │  MobileNetV2 / CNN  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Disease Prediction  │
                └──────────┬──────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          Healthy      Leaf Rust    Powdery Mildew
                           │
                           ▼
                ┌─────────────────────┐
                │ Disease Information │
                │ & Recommendations   │
                └─────────────────────┘
Ongoing project 
