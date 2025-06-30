# 😊 Facial Expression Classification Web App (7 Classes)

This Flask-based web application classifies facial expressions from uploaded images using a pre-trained deep learning model. It supports 7 emotion classes and stores each prediction in a MongoDB database along with its confidence score and timestamp.

---

## 📌 Features

- Classifies facial expressions into one of the following:
  - `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
- REST API for image classification
- MongoDB integration to store predictions
- HTML upload form for testing via browser
- Preprocessing pipeline using OpenCV

---

## 🧠 Model Information

- Model: Convolutional Neural Network (CNN)
- Format: `facial_model.h5`
- Input: 224x224 RGB image
- Output: Softmax probability across 7 emotion classes

---

## 🗂️ Project Structure
├── app.py # Main Flask application
├── model/
│ └── facial_model.h5 # Trained Keras model
├── templates/
│ └── index.html # Upload form (optional frontend)
├── static/ # Optional: for JS/CSS files
├── requirements.txt # Required Python packages
└── README.md # This file


---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/facial-expression-classifier.git
cd facial-expression-classifier
```

### 2. Create a Virtual Environment
```bash 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
Sample requirements.txt:
Flask
tensorflow
numpy
opencv-python
pymongo

### Running the App
Start MongoDB
Ensure MongoDB is running locally:
```bash
mongod --dbpath "C:/data/db"  # Update path if necessary
```

### Launch Flask App
```bash
python app.py
```

Visit: http://127.0.0.1:5000

