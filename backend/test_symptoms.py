"""
Test script to demonstrate symptom-based disease prediction
and show actual symptom names from the dataset
"""
import pickle
import pandas as pd
import numpy as np

# Load the trained model and symptom list
with open('disease_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('symptom_list.pkl', 'rb') as f:
    symptom_list = pickle.load(f)

# Load dataset to see actual disease-symptom combinations
df = pd.read_csv('../data/dataset.csv')
# Remove Unnamed columns and normalize prognosis names
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df['prognosis'] = df['prognosis'].str.strip()

print("=" * 60)
print("SYMPTOM-BASED DISEASE PREDICTION - TEST CASES")
print("=" * 60)

# Get some sample diseases and their exact symptoms from the dataset
diseases_to_show = ['Diabetes', 'Malaria', 'Pneumonia', 'Common Cold', 'Migraine']

for disease_name in diseases_to_show:
    # Find a row with this disease
    disease_rows = df[df['prognosis'] == disease_name]
    
    if len(disease_rows) > 0:
        # Get the first example
        sample_row = disease_rows.iloc[0]
        
        # Extract symptoms where value = 1
        active_symptoms = []
        for col in df.columns:
            if col != 'prognosis' and sample_row[col] == 1:
                active_symptoms.append(col)
        
        print(f"\n{disease_name.upper()}")
        print("-" * 60)
        print(f"Number of symptoms: {len(active_symptoms)}")
        print("\nSymptoms to enter (copy these):")
        
        # Show in user-friendly format
        for symptom in active_symptoms:
            print(f"  - {symptom.replace('_', ' ').title()}")
        
        # Test prediction
        feature_vector = [1 if s in active_symptoms else 0 for s in symptom_list]
        prediction = model.predict([feature_vector])[0]
        probabilities = model.predict_proba([feature_vector])[0]
        confidence = max(probabilities) * 100
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_diseases = [model.classes_[i] for i in top_3_indices]
        top_3_probs = [probabilities[i] * 100 for i in top_3_indices]
        
        print(f"\nPredicted: {prediction}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Match: {'YES' if prediction == disease_name else 'NO'}")
        print(f"\nTop 3 predictions:")
        for i, (dis,  prob) in enumerate(zip(top_3_diseases, top_3_probs), 1):
            print(f"  {i}. {dis}: {prob:.2f}%")

print("\n" + "=" * 60)
print("TOTAL SYMPTOMS IN DATABASE:", len(symptom_list))
print("TOTAL DISEASES:", len(df['prognosis'].unique()))
print("=" * 60)

# Show what happens with partial symptoms
print("\n" + "=" * 60)
print("TESTING WITH PARTIAL SYMPTOMS")
print("=" * 60)

# Test Diabetes with only 3 symptoms instead of all
diabetes_row = df[df['prognosis'] == 'Diabetes'].iloc[0]
all_diabetes_symptoms = [col for col in df.columns if col != 'prognosis' and diabetes_row[col] == 1]

# Take only first 3 symptoms
partial_symptoms = all_diabetes_symptoms[:3]

print(f"\nUsing only {len(partial_symptoms)} out of {len(all_diabetes_symptoms)} Diabetes symptoms:")
for s in partial_symptoms:
    print(f"  - {s.replace('_', ' ').title()}")

feature_vector = [1 if s in partial_symptoms else 0 for s in symptom_list]
prediction = model.predict([feature_vector])[0]
probabilities = model.predict_proba([feature_vector])[0]
confidence = max(probabilities) * 100

print(f"\nPredicted: {prediction}")
print(f"Confidence: {confidence:.2f}%")
print(f"Note: Confidence is {'LOWER' if confidence < 80 else 'HIGHER'} with partial symptoms")

print("\n" + "=" * 60)
print("TIP: For best results, enter 4-6 symptoms that match the disease")
print("=" * 60)
