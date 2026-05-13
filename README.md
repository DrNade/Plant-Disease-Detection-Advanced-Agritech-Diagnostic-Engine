# Plant-Disease-Detection-Advanced-Agritech-Diagnostic-Engine
# 🌱 InovaPlant AI - Advanced Agritech Diagnostic Engine

![InovaPlant AI Banner](https://img.shields.io/badge/AI_Engine-TensorFlow_2.x-10b981?style=for-the-badge&logo=tensorflow)
![Backend](https://img.shields.io/badge/Backend-Flask-000000?style=for-the-badge&logo=flask)
![Frontend](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-e34f26?style=for-the-badge&logo=html5)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

**InovaPlant AI** is a professional, web-based agricultural diagnostic tool powered by deep learning. It uses an **Ensemble Learning Approach (Dual Model Architecture)** to analyze plant leaf images, accurately identify 38 different plant diseases, and provide instant medical reports including symptoms, treatments, and local remedies.

---

## ✨ Key Features

* **🧠 Dual-Engine AI (Ensemble Learning):** Combines predictions from two different Convolutional Neural Networks (CNNs) to ensure maximum accuracy and eliminate false positives.
* **📷 Live Camera & Gallery Support:** Seamlessly capture leaf images directly from your mobile/webcam or upload from your device.
* **🩺 Professional Diagnostic Report:** Generates a complete report detailing the plant name, disease, exact symptoms, and recommended treatments (e.g., SSP Fertilizers, Fungicides).
* **📊 Confidence Analysis Graph:** Visualizes the AI's confidence level in real-time using Chart.js.
* **⏱️ Real-time Processing Tracking:** Displays the exact time taken for image processing and inference.
* **💾 Local History Management:** Automatically saves your recent scans in the browser using LocalStorage.
* **🌐 Dynamic Research Integration:** One-click automated Google Search generation for disease treatment and prevention in Urdu/English without relying on paid APIs.

---

## 🛠️ Tech Stack

* **Machine Learning:** TensorFlow / Keras (MobileNetV2 / Custom CNN architecture)
* **Backend:** Python, Flask, Werkzeug
* **Frontend:** HTML5, CSS3, Vanilla JavaScript, Chart.js, Lucide Icons
* **Data Processing:** NumPy, Pillow (PIL)

---

## 📂 Project Structure

```text
InovaPlant-AI/
│
├── app.py                      # Main Flask application and AI logic
├── best_phase1.h5              # AI Model 1 (Primary Brain)
├── plant_disease_model old.h5  # AI Model 2 (Secondary Brain)
├── labels.txt                  # 38 Class definitions
│
├── templates/
│   └── index.html              # Premium User Interface (UI)
│
├── uploads/                    # Temporary storage for scanned images
└── requirements.txt            # Python dependencies

🚀 Installation & Setup Guide

Follow these steps to run the InovaPlant AI Server locally on your machine.

1. Clone the repository
Bash

git clone [https://github.com/your-username/InovaPlant-AI.git](https://github.com/your-username/InovaPlant-AI.git)
cd InovaPlant-AI

2. Create a Virtual Environment (Recommended)
Bash

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install Dependencies
Bash

pip install -r requirements.txt

4. Run the Flask Server
Bash

python app.py

The server will start on http://127.0.0.1:5000/. Open this link in your browser.
🎯 Supported Crops & Diseases (38 Classes)

The model is highly trained on multiple crops including, but not limited to:

    Apple: Scab, Black Rot, Cedar Apple Rust

    Corn (Maize): Common Rust, Northern Leaf Blight, Cercospora

    Tomato & Potato: Early Blight, Late Blight, Septoria Leaf Spot, Spider Mites

    Others: Grape, Orange, Peach, Pepper, Strawberry, Soybean, Squash.

👨‍💻 Developed By

Muhammad Nadeem Aslam IT Infrastructure & AI Development

    Brand: InovaTech

Disclaimer: This tool is designed to assist farmers and agronomists. It is advised to consult a certified agricultural professional before applying chemical treatments based solely on AI results.


---

### 2. `requirements.txt` File

GitHub par jab bhi koi Python project rakha jata hai, toh uske sath ek `requirements.txt` file lazmi hoti hai taake doosre log command chala kar zaroori software install kar sakein. Apne main folder mein ek file banayein jiska naam `requirements.txt` ho aur usme yeh paste kar dein:

```text
Flask==3.0.0
flask-cors==4.0.0
tensorflow==2.15.0
numpy==1.26.4
Pillow==10.2.0
Werkzeug==3.0.1
