import numpy as np
from model import ImagePredictor
import os

# Initialize predictor
predictor = ImagePredictor()

print("=" * 60)
print("IMAGE MODEL DIAGNOSTIC TEST")
print("=" * 60)
print(f"\nModel loaded: {predictor.model is not None}")
print(f"Classes: {predictor.classes}")
print("\n" + "-" * 60)

# Check if uploads directory exists
uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
if os.path.exists(uploads_dir):
    print(f"\nChecking images in: {uploads_dir}")
    image_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    if image_files:
        print(f"Found {len(image_files)} images\n")
        
        for img_file in image_files[:5]:  # Test first 5 images
            img_path = os.path.join(uploads_dir, img_file)
            print(f"\nTesting: {img_file}")
            print("-" * 40)
            
            try:
                result = predictor.predict(img_path)
                print(f"  Disease: {result['disease']}")
                print(f"  Confidence: {result['confidence']}%")
                
                # Also check raw model output if model exists
                if predictor.model is not None:
                    from PIL import Image
                    img = Image.open(img_path).convert('RGB').resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    predictions = predictor.model.predict(img_array, verbose=0)
                    print(f"  Raw predictions: {predictions[0]}")
                    print(f"  Class 0 (NORMAL): {predictions[0][0]:.4f}")
                    print(f"  Class 1 (PNEUMONIA): {predictions[0][1]:.4f}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        print("No images found in uploads directory")
        print("\nPlease upload some test images through the web interface first.")
else:
    print(f"Uploads directory not found: {uploads_dir}")
    print("Please upload some images through the web interface first.")

print("\n" + "=" * 60)
