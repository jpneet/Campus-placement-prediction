# Campus-placement-prediction
🎓 Campus Placement Prediction using Machine Learning
📌 Project Overview

This project focuses on predicting campus placement outcomes of students using machine learning techniques. By analyzing academic performance, skill indicators, and extracurricular involvement, the model estimates whether a student is likely to be placed or not placed.

The pipeline covers the entire ML lifecycle:
data preprocessing → visualization → model training → evaluation → hyperparameter tuning → explainability.

📊 Dataset

Source: placedata v2.0 synthetic.csv

Type: Synthetic dataset

Target Variable: PlacementStatus (Placed / Not Placed)

Features include:

Academic scores (CGPA, SSC, HSC)

Aptitude test scores

Soft skills rating

Internships, projects, workshops

Placement training & extracurricular activities

⚙️ Technologies & Libraries

Programming Language: Python

Core Libraries:

NumPy, Pandas

Matplotlib, Seaborn

Machine Learning:

Scikit-learn

XGBoost

Imbalanced-learn (SMOTE)

Explainability:

SHAP

🧹 Data Preprocessing

Removed irrelevant identifiers (StudentID)

Label-encoded categorical features

Standardized numerical features using StandardScaler

Handled class imbalance using SMOTE

Engineered new features:

CGPA_Aptitude = CGPA × AptitudeTestScore

SoftSkills_Extracurricular = SoftSkillsRating + ExtracurricularActivities

📈 Exploratory Data Analysis

Correlation heatmap

Placement distribution analysis

Count plots for categorical features

Boxplots for numerical features vs placement status

🤖 Machine Learning Models Implemented

The following models were trained and evaluated:

Logistic Regression

Support Vector Machine (SVM)

Random Forest

Decision Tree

K-Nearest Neighbors (KNN)

AdaBoost

Gradient Boosting

XGBoost

🗳️ Ensemble Learning

Voting Classifier (Soft Voting) using:

Logistic Regression

SVM

AdaBoost

Custom weights applied to improve performance

Threshold tuning performed for optimal accuracy

📐 Model Evaluation Metrics

Accuracy

ROC-AUC Score

Confusion Matrix

ROC Curves

Train vs Test Accuracy comparison

Stratified 5-Fold Cross-Validation (ROC-AUC)

🔍 Hyperparameter Tuning

GridSearchCV for Random Forest

RandomizedSearchCV for XGBoost

Optimized using ROC-AUC as the primary metric

🧠 Model Explainability (XAI)

Implemented SHAP (SHapley Additive exPlanations) to:

Identify globally important features

Visualize feature impact on placement prediction

Used appropriate explainers:

TreeExplainer

LinearExplainer

KernelExplainer (for complex models)

📊 Visual Performance Analysis

Train vs Test confusion matrices

ROC curves for train and test sets

Comparative bar chart of model accuracies

🚀 Key Outcomes

Ensemble models outperform individual classifiers

CGPA, aptitude score, soft skills, and internships play a major role in placement

SHAP enhances interpretability, making the model suitable for academic and real-world use

▶️ How to Run

Clone the repository

Install dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn shap


Place the dataset file in the project directory

Run the notebook:

jupyter notebook Campus_placement.ipynb

📌 Future Enhancements

Deployment using Flask or FastAPI

Real-time prediction dashboard

Feature selection using SHAP-based pruning

Integration with student information systems

👤 Author

Japneet Singh
Machine Learning Enthusiast | Data Science | AI
