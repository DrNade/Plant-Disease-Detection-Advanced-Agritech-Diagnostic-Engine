import os
import time
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense

class FixedDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==========================================
# 🧠 DUAL MODEL LOADING (ENSEMBLE)
# ==========================================
MODEL_1_PATH = 'plant_disease_model.h5'
MODEL_2_PATH = 'plant_disease_model1.h5'

print("Loading Brain 1 & 2...")
model1 = tf.keras.models.load_model(MODEL_1_PATH, custom_objects={'Dense': FixedDense}, compile=False)
model2 = tf.keras.models.load_model(MODEL_2_PATH, custom_objects={'Dense': FixedDense}, compile=False)
print("Dual AI Specialist Ready!")

# ==========================================
# 📖 LABELS.TXT SE ASLI NAAM UTHANA
# ==========================================
def load_labels(label_path):
    labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                # Agar labels '0 Apple' format mein hain tou number hata dega
                parts = line.split(' ', 1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels.append(parts[1])
                else:
                    labels.append(line)
        print(f"✅ {len(labels)} Asli Labels load ho gaye hain!")
        return labels
    except Exception as e:
        print("❌ labels.txt file nahi mili! Error:", e)
        return []

class_names = load_labels('labels.txt')
# ==========================================

def get_disease_details(disease_name):
    d_lower = disease_name.lower()
    if "healthy" in d_lower:
        return "Patte bilkul saaf aur sehatmand hain.", "SSP Fertilizer ka munasib istemal jari rakhein."
    elif "rust" in d_lower:
        return "Patton par zang (rust) jese peele ya narangi powder wale dhabbe hain.", "Zang-aalood patte nikal dein aur Sulfur-based fungicide spray karein."
    elif "early_blight" in d_lower or "target_spot" in d_lower:
        return "Patton par bhuray dhabbe aur target board nishan hain.", "Kharab patte tor dein aur Mancozeb spray karein."
    else:
        return "Patton par fungus ya bacteria ke asraat hain.", "Broad-spectrum Fungicide ka spray karein."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_time = time.time()
        file = request.files.get('file')
        if not file: return "No file", 400
            
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)
        
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img) / 255.0  # Accuracy Fix
        x = np.expand_dims(x, axis=0)
        
        # Ensemble Prediction
        p1 = model1.predict(x)
        p2 = model2.predict(x)
        final_preds = (p1 + p2) / 2.0
        
        idx = np.argmax(final_preds)
        conf = float(np.max(final_preds))
        
        full_name = class_names[idx]
        plant = full_name.split('___')[0].replace('_', ' ')
        disease = full_name.split('___')[1].replace('_', ' ')
        symptoms, remedy = get_disease_details(disease)
        
        return render_template('index.html', plant=plant, disease=disease, confidence=conf, 
                               filename=filename, proc_time=round(time.time()-start_time, 2),
                               status="HEALTHY" if "healthy" in disease.lower() else "DISEASED",
                               symptoms=symptoms, prevention=remedy)
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)