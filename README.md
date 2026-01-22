# Campus Placement Prediction

This project is a Machine Learning pipeline to predict campus placement outcomes based on student data. It involves data preprocessing, feature engineering, visualization, model training, hyperparameter tuning, ensemble methods, and explainability using SHAP.

---

## Table of Contents

- [Dataset](#dataset)  
- [Libraries Used](#libraries-used)  
- [Data Preprocessing](#data-preprocessing)  
- [Feature Engineering](#feature-engineering)  
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
- [Modeling](#modeling)  
- [Model Evaluation](#model-evaluation)  
- [Ensemble Learning](#ensemble-learning)  
- [Hyperparameter Tuning](#hyperparameter-tuning)  
- [Explainability](#explainability)  
- [Visualization](#visualization)  

---

## Dataset

- The dataset `placedata v2.0 synthetic.csv` contains synthetic data for students including features like CGPA, soft skills, internships, projects, workshops, aptitude test scores, extracurricular activities, and placement training.
- Target variable: `PlacementStatus` (Placed / Not Placed)
- Unique identifier `StudentID` is dropped during preprocessing.

---

## Libraries Used

- `numpy`  
- `pandas`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  
- `xgboost`  
- `imblearn` (SMOTE)  
- `shap`  

---

## Data Preprocessing

- Dropped `StudentID` column to remove irrelevant identifiers.
- Encoded categorical variables using `LabelEncoder`.
- Split dataset into features `X` and target `y`.
- Train-test split: 80%-20% with stratification to avoid class imbalance.
- Feature scaling using `StandardScaler`.
- Addressed class imbalance with SMOTE.

---

## Feature Engineering

- Created combined features for better predictive power:  
  - `CGPA_Aptitude = CGPA * AptitudeTestScore`  
  - `SoftSkills_Extracurricular = SoftSkillsRating + ExtracurricularActivities`  

---

## Exploratory Data Analysis (EDA)

- Descriptive statistics for numerical features.
- Correlation heatmap for all features.
- Countplots for `PlacementStatus` and categorical features (`PlacementTraining`, `ExtracurricularActivities`).
- Boxplots for numerical features vs placement status.

---

## Modeling

Models trained and evaluated:

1. Support Vector Machine (SVM)  
2. Logistic Regression  
3. Random Forest  
4. Decision Tree  
5. K-Nearest Neighbors (KNN)  
6. XGBoost  
7. AdaBoost  
8. Gradient Boosting  

- Train and test accuracies calculated.
- Confusion matrices and ROC-AUC scores evaluated.

---

## Model Evaluation

- Performance metrics include:
  - Accuracy
  - ROC-AUC score
  - Confusion Matrix
- Stratified K-Fold cross-validation (5-fold) for ROC-AUC evaluation.

---

## Ensemble Learning

- Soft Voting Classifier combining:
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
- Voting Classifier with weights and soft voting improves overall prediction.
- Threshold tuning performed to maximize accuracy.

---

## Hyperparameter Tuning

- **Random Forest**: GridSearchCV to optimize `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
- **XGBoost**: RandomizedSearchCV for `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.

---

## Explainability

- SHAP used to explain feature importance for:
  - Individual models
  - Voting Classifier (Logistic Regression component)
- Summary plots and bar plots generated for global interpretability.

---

## Visualization

- Confusion matrices and ROC curves for train/test sets.
- Comparison bar plots for train vs test accuracy across models.
- Visualizations aid in understanding model performance and feature contributions.

---


Author

Japneet Singh
Computer Science & Engineering | ML/AI Enthusiast




