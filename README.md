# Heart Disease Predictor (78% Accuracy)

**Clinical-grade cardiovascular risk prediction using the UCI Cleveland Heart Disease dataset and Scikit-Learn!**

## 📌 Results


| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **78.2%** |
| **Data Points** | 270 Patients |
| **Features** | 13 Clinical Attributes + Target |
| **Model** |   XGBOOST + ColumnTransformer Pipeline |

## 📁 Dataset
[Kaggle Heart Disease Dataset](https://www.kaggle.com)  
**14 Key Attributes** including Age, Sex, Cholesterol, and Thallium stress test results. 

## 🚀 Key Features
*   **Automated Pipeline**: Handles categorical encoding (Sex, Chest Pain) and numerical scaling (BP, HR) simultaneously.
*   **Robust Preprocessing**: Uses `StandardScaler` and `OneHotEncoder` to prevent data leakage.
*   **Portable Model**: Serialized with `joblib` for instant production deployment.

## 🛠️ Installation
```bash
pip install pandas scikit-learn joblib
