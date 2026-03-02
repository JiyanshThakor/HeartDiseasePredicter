import joblib as jb
import pandas as pd

model = jb.load("HeartDiseasePredicter.pkl")

patient_data = {
    'Age': [67], 
    'Sex': [1],                     # Male (Higher statistical risk in this set)
    'Chest pain type': [4],         # Asymptomatic (often the most dangerous type)
    'BP': [160],                    # Hypertension
    'Cholesterol': [286],           # High Cholesterol
    'FBS over 120': [1],            # Diabetic/High Blood Sugar
    'EKG results': [2],             # Left ventricular hypertrophy
    'Max HR': [108],                # Low Max Heart Rate (Strong indicator)
    'Exercise angina': [1],         # Yes (Pain during exercise)
    'ST depression': [1.5],         # Significant ST depression
    'Slope of ST': [2],             # Flat/Downsloping
    'Number of vessels fluro': [3], # 3 vessels colored (Very high risk)
    'Thallium': [7]                 # Reversible defect (Classic disease marker)
}

df_patient = pd.DataFrame(patient_data)

categorical_cols = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Thallium']
for col in categorical_cols:
    if col in df_patient.columns:
        df_patient[col] = df_patient[col].astype(str)

prob = model.predict_proba(df_patient)[0] 


print(f"Probability of No: {prob[1]*100:.2f}%")
print(f"Probability of Yes: {prob[0]*100:.2f}%")