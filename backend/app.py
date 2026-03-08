from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from model import SymptomPredictor, ImagePredictor

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize predictors
print("[INFO] Initializing predictors...")
symptom_predictor = SymptomPredictor()
image_predictor = ImagePredictor()
print("[INFO] Predictors initialized successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/symptom-checker')
def symptom_checker():
    return render_template('symptom_checker.html')

@app.route('/image-analysis')
def image_analysis():
    return render_template('image_analysis.html')

@app.route('/predict-symptoms', methods=['POST'])
def predict_symptoms():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        # Get prediction
        result = symptom_predictor.predict(symptoms)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"[ERROR] Symptom prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict-image', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Auto-detect across all models (no image_type needed)
            result = image_predictor.predict(filepath)
            
            # Optional: Clean up uploaded file after prediction
            # os.remove(filepath)
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or GIF'}), 400
    
    except Exception as e:
        print(f"[ERROR] Image prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-symptoms', methods=['GET'])
def get_symptoms():
    try:
        symptoms = symptom_predictor.get_all_symptoms()
        return jsonify({'symptoms': symptoms})
    except Exception as e:
        print(f"[ERROR] Failed to get symptoms: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-image-categories', methods=['GET'])
def get_image_categories():
    try:
        categories = image_predictor.get_available_categories()
        return jsonify({'categories': categories})
    except Exception as e:
        print(f"[ERROR] Failed to get image categories: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("[STARTUP] Starting AI Disease Recognition System")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
