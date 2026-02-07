# 🎓 Campus Placement Prediction | Machine Learning

## Overview
End-to-end **binary classification machine learning pipeline** to predict campus placement outcomes using academic performance, aptitude, skills, and extracurricular indicators.  
The project emphasizes **generalization, interpretability, and robust evaluation**, not accuracy alone.

---

## Dataset
- **Source:** `placedata_v2.0_synthetic.csv`
- **Target:** `PlacementStatus` (Placed / Not Placed)
- **Features:** CGPA, aptitude scores, soft skills, extracurriculars, training
- **Preprocessing:** Label encoding, one-hot encoding, scaling, leakage-safe splits

---

## Feature Engineering
- **CGPA_Aptitude** = `CGPA × AptitudeTestScore`
- **SoftSkills_Extracurricular** = `SoftSkillsRating + ActivityParticipation`

---

## Models Evaluated
- Logistic Regression (**final**)
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)
- XGBoost
- AdaBoost
- Gradient Boosting

**Validation:** Stratified 5-Fold Cross-Validation (ROC-AUC)

---

## Results
**Best Overall Model:** **Logistic Regression**

- **ROC-AUC (CV):** 0.8769 ± 0.0132  
- **Test Accuracy:** **80.90%**  
- **F1-Score:** **0.7729**

Balanced precision–recall, minimal overfitting, high interpretability, and deployment readiness.

---

## Tech Stack
- **Language:** Python
- **Libraries:** NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Google Colab

---

## Run Instructions
```bash
# Open the notebook
Campus_Placement.ipynb

# Load dataset
placedata_v2.0_synthetic.csv

```

# Run all cells to train and evaluate models

Inference

Apply identical preprocessing → generate engineered features → scale features → predict using the trained Logistic Regression model.

Author

Japneet Singh
B.Tech (Prefinal Year) | Machine Learning & Data Science
