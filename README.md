# Diabetes 30-Day Readmission Prediction

## Project Description
This project uses machine learning to predict whether a diabetic
patient will be readmitted to the hospital within 30 days.

## Dataset
Diabetes 130-US Hospitals for Years 1999-2008 (UCI Machine Learning Repository)
Link: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

The dataset is not included in this repository.
Download it and place diabetic_data.csv in the project folder.

## Algorithms
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- Hist Gradient Boosting
- K-Nearest Neighbors
- Naive Bayes
- Linear SVM
- LDA
- Neural Network (MLP)

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Project Files
- train_models.py: trains Logistic Regression and Random Forest
- train_all_models.py: trains and compares all 12 models
- outputs/: results of the first script
- outputs_all/: results and charts of the second script
- requirements.txt: Python packages needed

## How to Run
1. Install Python.
2. Install the packages:
   pip install -r requirements.txt
3. Put diabetic_data.csv in the project folder.
4. Run:
   python train_all_models.py

## Results
The data is imbalanced: only about 11% of patients are readmitted within 30 days.
- 11 of 12 models reach about 88.8% accuracy on the 20% test set.
- Best ROC-AUC: Hist Gradient Boosting (0.68).
- Recall for readmitted patients is very low for most models.
- Train and test accuracy are almost equal, so there is no overfitting.
Full table: outputs_all/all_models_comparison.csv

## Contributors
- Original author: Wanya
- Collaborator: (add your partner's name later)