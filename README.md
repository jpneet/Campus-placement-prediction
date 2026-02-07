🎓 Campus Placement Prediction | Machine Learning
Overview

End-to-end binary classification ML pipeline to predict campus placement outcomes using academic performance, aptitude, skills, and extracurricular indicators.
Focus on generalization, interpretability, and robust evaluation, not accuracy alone.

Dataset

Source: placedata_v2.0_synthetic.csv

Target: PlacementStatus (Placed / Not Placed)

Features: CGPA, aptitude scores, soft skills, extracurriculars, training

Preprocessing: Label encoding, one-hot encoding, scaling, leakage-safe splits

Feature Engineering

CGPA_Aptitude = CGPA × AptitudeTestScore

SoftSkills_Extracurricular = SoftSkillsRating + ActivityParticipation

Models Evaluated

Logistic Regression (final) · SVM · Random Forest · Decision Tree · KNN · XGBoost · AdaBoost · Gradient Boosting
Validation: Stratified 5-Fold Cross-Validation (ROC-AUC)

Results

Best Overall Model: Logistic Regression

ROC-AUC (CV): 0.8769 ± 0.0132

Test Accuracy: 80.90%

F1-Score: 0.7729

Balanced precision–recall, minimal overfitting, high interpretability, deployment-ready.

Tech Stack

Python · NumPy · Pandas · Scikit-learn · XGBoost · Matplotlib · Seaborn
Jupyter / Google Colab

Run Instructions

```bash
1. Open Campus_Placement.ipynb
2. Load placedata_v2.0_synthetic.csv
3. Run all cells
```
Inference

Apply identical preprocessing → generate engineered features → scale → predict using trained Logistic Regression model.

Author

Japneet Singh
B.Tech (Prefinal Year) | ML & Data Science
