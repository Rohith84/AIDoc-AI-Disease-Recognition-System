import os
import sys
import numpy as np
from model import ImagePredictor
import json

def diagnostic():
    print("=" * 60)
    print("IMAGE PREDICTION DIAGNOSTIC")
    print("=" * 60)
    
    # Initialize predictor
    predictor = ImagePredictor()
    
    print(f"\n[STATUS] Available categories: {predictor.available_categories}")
    print(f"[STATUS] Models loaded: {list(predictor.models.keys())}")
    print(f"[STATUS] Router loaded: {predictor.router_model is not None}")
    
    if not predictor.models:
        print("\n[WARNING] NO TRAINED MODELS LOADED! System will use demo logic.")
    
    # Check uploads
    uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
    if not os.path.exists(uploads_dir):
        print(f"\n[ERROR] Uploads directory not found: {uploads_dir}")
        return
        
    image_files = sorted([f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], reverse=True)
    
    if not image_files:
        print("\n[INFO] No images found in uploads.")
        return
        
    print(f"\n[INFO] Testing latest {min(3, len(image_files))} images:")
    
    for img_file in image_files[:3]:
        img_path = os.path.join(uploads_dir, img_file)
        print(f"\n--- Testing: {img_file} ---")
        try:
            result = predictor.predict(img_path)
            print(f"  Detected Type: {result.get('image_type', 'Unknown')}")
            print(f"  Predicted Disease: {result.get('disease', 'Unknown')}")
            print(f"  Confidence: {result.get('confidence', 0)}%")
            
            # Diagnostic: check if it's the demo logic
            # In demo logic, TB confidence is between 72 and 86
            if result.get('disease') == 'Tuberculosis' and 72 <= result.get('confidence') <= 86:
                print("  [DIAGNOSTIC] Result matches demo logic range for TB.")
                
        except Exception as e:
            print(f"  [ERROR] {e}")

if __name__ == "__main__":
    diagnostic()
