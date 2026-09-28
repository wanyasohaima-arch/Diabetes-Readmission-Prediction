"""
Diabetes 30-Day Readmission - Lab Task
Preprocess data, split 80/20, train Logistic Regression + Random Forest,
compare metrics, plot confusion matrices, save trained models as .pkl
"""

import os
import pandas as pd
import numpy as np
import joblib
import json

os.makedirs("outputs", exist_ok=True)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# ---------------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("diabetic_data.csv")
print("Original shape:", df.shape)

# ---------------------------------------------------------------
# STEP 2: CLEAN DATA
# ---------------------------------------------------------------
# "?" means missing in this dataset -> convert to real NaN
df = df.replace("?", np.nan)

# Drop columns that are IDs, almost entirely missing, or too high-cardinality
# to be simple, reliable web-form inputs (diag_1/2/3 are ICD-9 codes with
# hundreds of distinct values each).
drop_cols = [
    "encounter_id", "patient_nbr",   # identifiers, not predictive
    "weight", "payer_code", "medical_specialty",  # >40% missing
    "diag_1", "diag_2", "diag_3",    # very high-cardinality diagnosis codes
]
df = df.drop(columns=drop_cols)

# Remove the 3 rows with unusable gender value
df = df[df["gender"] != "Unknown/Invalid"]

# Fill remaining small amount of missing categorical data with "Unknown"
df["race"] = df["race"].fillna("Unknown")

# ---------------------------------------------------------------
# STEP 3: CREATE BINARY TARGET
# ---------------------------------------------------------------
# <30  -> 1 (readmitted within 30 days)
# >30 / NO -> 0 (not readmitted within 30 days)
df["target"] = (df["readmitted"] == "<30").astype(int)
df = df.drop(columns=["readmitted"])

print("\nTarget balance:")
print(df["target"].value_counts(normalize=True).rename("proportion"))

X = df.drop(columns=["target"])
y = df["target"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
print(f"\n{len(categorical_cols)} categorical columns, {len(numeric_cols)} numeric columns")

# ---------------------------------------------------------------
# STEP 4: TRAIN / TEST SPLIT (80/20, stratified so both classes
# keep the same ratio in train and test)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------
# STEP 5: PREPROCESSING PIPELINE
# (one-hot encode categoricals, scale numeric features)
# Wrapped in a ColumnTransformer so the SAME transformation is
# reused later inside the web app -- no separate "refit" needed.
# ---------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# ---------------------------------------------------------------
# STEP 6: DEFINE THE TWO MODELS
# Random Forest depth/leaf size are limited on purpose -- an
# unrestricted forest memorizes the training data (overfits).
# ---------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]),
    "Random Forest": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,          # caps tree complexity -> reduces overfitting
            min_samples_leaf=20,   # requires enough samples per leaf
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),
}

results = {}

for name, pipe in models.items():
    print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")
    pipe.fit(X_train, y_train)

    # ---- 5-fold cross-validation on the TRAINING set only ----
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"5-fold CV accuracy (train): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # ---- performance on train vs test (checks for overfitting) ----
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    test_pred = pipe.predict(X_test)
    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, test_pred)

    gap = train_acc - test_acc
    print(f"Train accuracy: {train_acc:.4f}   Test accuracy: {test_acc:.4f}   Gap: {gap:.4f}")
    if gap > 0.05:
        print("  -> WARNING: train/test gap > 5%, possible overfitting")
    else:
        print("  -> OK: train and test accuracy are close, no significant overfitting")

    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_proba)

    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-score:  {f1:.4f}")
    print(f"Test ROC-AUC:   {roc_auc:.4f}")

    results[name] = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

    # ---- confusion matrix plot ----
    cm = confusion_matrix(y_test, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No 30d Readmit", "30d Readmit"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    fname = f"outputs/confusion_matrix_{name.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)

    # ---- save trained model ----
    model_fname = f"outputs/{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(pipe, model_fname)
    print(f"Saved model -> {model_fname}")

# ---------------------------------------------------------------
# STEP 7: SIDE-BY-SIDE COMPARISON TABLE
# ---------------------------------------------------------------
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df[["test_accuracy", "precision", "recall", "f1", "roc_auc", "train_accuracy", "cv_accuracy_mean"]]
comparison_df.columns = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Train Acc", "CV Acc (train)"]
print("\n" + "=" * 70)
print("MODEL COMPARISON (test set)")
print("=" * 70)
print(comparison_df.round(4))

comparison_df.round(4).to_csv("outputs/model_comparison.csv")

with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save the list of columns the pipeline expects, for the web app form later
with open("outputs/feature_columns.json", "w") as f:
    json.dump({
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "categorical_options": {c: sorted(X[c].dropna().unique().tolist()) for c in categorical_cols},
    }, f, indent=2)

print("\nAll done. Files saved in outputs/")