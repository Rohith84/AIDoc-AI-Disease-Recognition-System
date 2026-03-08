import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SymptomPredictor

def test_mapper():
    print("Initializing SymptomPredictor...")
    predictor = SymptomPredictor()
    
    test_cases = [
        # Test Case 1: Direct containment (Severe Headache -> headache)
        ["Severe Headache", "High Fever"],
        
        # Test Case 2: Fuzzy match (High Temp -> high_fever)
        ["High Temp", "Coughing"],
        
        # Test Case 3: Mixed known and unknown
        ["headache", "Pain in Chest", "UnknownXYZ"],
        
        # Test Case 4: No match
        ["TotallyRandomSymptom123"]
    ]
    
    print("\n" + "="*50)
    print("TESTING INTELLIGENT SYMPTOM MAPPING")
    print("="*50)
    
    for i, symptoms in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: Input -> {symptoms}")
        result = predictor.predict(symptoms)
        
        print(f"Prediction: {result['disease']}")
        print(f"Confidence: {result['confidence']}%")
        
        if 'mapped_symptoms' in result and result['mapped_symptoms']:
            print("MAPPING SUCCESS:")
            for orig, mapped in result['mapped_symptoms'].items():
                print(f"  - '{orig}' mapped to '{mapped}'")
        else:
            print("No mapping occurred.")
            
        if 'warning' in result:
            print(f"Warning: {result['warning']}")
            
        print("-"*30)

if __name__ == "__main__":
    test_mapper()
