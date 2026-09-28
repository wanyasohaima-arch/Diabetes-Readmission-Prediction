"""
Diabetes 30-Day Readmission - Train ALL models and compare them
Same cleaning + 80/20 split as before, but now many models.
Saves every model as .pkl, plus a comparison table and confusion matrices.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------
# SETTINGS (you can change these)
# ---------------------------------------------------------------
RANDOM_STATE = 42
BALANCE_CLASSES = False   # False = normal (high accuracy, low recall)
                          # True  = balanced (better recall, lower accuracy)
OUT = "outputs_all"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/models", exist_ok=True)
os.makedirs(f"{OUT}/confusion_matrices", exist_ok=True)

CW = "balanced" if BALANCE_CLASSES else None

# ---------------------------------------------------------------
# STEP 1: LOAD + CLEAN (same as before)
# ---------------------------------------------------------------
df = pd.read_csv("diabetic_data.csv")
print("Original shape:", df.shape)

df = df.replace("?", np.nan)
df = df.drop(columns=[
    "encounter_id", "patient_nbr",
    "weight", "payer_code", "medical_specialty",
    "diag_1", "diag_2", "diag_3",
])
df = df[df["gender"] != "Unknown/Invalid"]
df["race"] = df["race"].fillna("Unknown")

# target: <30 days = 1, everything else = 0
df["target"] = (df["readmitted"] == "<30").astype(int)
df = df.drop(columns=["readmitted"])

X = df.drop(columns=["target"])
y = df["target"]

categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
numeric_cols = X.select_dtypes(include="number").columns.tolist()
print(f"{len(categorical_cols)} categorical columns, {len(numeric_cols)} numeric columns")

# ---------------------------------------------------------------
# STEP 2: SPLIT 80/20
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train)}   Test: {len(X_test)}")


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])


def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else x


# ---------------------------------------------------------------
# STEP 3: LIST OF ALL MODELS
# (settings are kept modest on purpose so they do not overfit)
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight=CW, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=50, class_weight=CW, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=20,
        class_weight=CW, n_jobs=-1, random_state=RANDOM_STATE),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=20,
        class_weight=CW, n_jobs=-1, random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=RANDOM_STATE),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=200,
        class_weight=CW, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=25, n_jobs=-1),
    "Naive Bayes": GaussianNB(),
    "Linear SVM": LinearSVC(
        C=0.1, class_weight=CW, max_iter=5000, random_state=RANDOM_STATE),
    "LDA": LinearDiscriminantAnalysis(),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(32,), alpha=0.01, early_stopping=True,
        max_iter=200, random_state=RANDOM_STATE),
}

# optional: XGBoost / LightGBM if the person has them installed
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE)
except Exception:
    print("(XGBoost not installed - skipping it. Optional: pip install xgboost)")

# ---------------------------------------------------------------
# STEP 4: TRAIN + TEST EVERY MODEL
# ---------------------------------------------------------------
rows = []
matrices = {}
trained = {}

for name, clf in models.items():
    print(f"\n--- Training: {name} ---")
    steps = [("preprocessor", make_preprocessor())]
    if name in ("Naive Bayes", "LDA"):
        steps.append(("dense", FunctionTransformer(to_dense)))
    steps.append(("classifier", clf))
    pipe = Pipeline(steps)

    t0 = time.time()
    try:
        pipe.fit(X_train, y_train)
    except Exception as e:
        print(f"   FAILED: {e}")
        continue

    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)

    # score for ROC-AUC (some models have no predict_proba)
    if hasattr(pipe, "predict_proba"):
        score = pipe.predict_proba(X_test)[:, 1]
    else:
        score = pipe.decision_function(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    row = {
        "Model": name,
        "Train Acc": train_acc,
        "Test Acc": test_acc,
        "Gap": train_acc - test_acc,
        "Precision": precision_score(y_test, test_pred, zero_division=0),
        "Recall": recall_score(y_test, test_pred, zero_division=0),
        "F1": f1_score(y_test, test_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, score),
        "Time (s)": time.time() - t0,
    }
    rows.append(row)
    trained[name] = pipe
    matrices[name] = confusion_matrix(y_test, test_pred)

    print(f"   Test Acc {row['Test Acc']:.4f} | Train Acc {row['Train Acc']:.4f} | "
          f"Precision {row['Precision']:.4f} | Recall {row['Recall']:.4f} | "
          f"F1 {row['F1']:.4f} | AUC {row['ROC-AUC']:.4f} | {row['Time (s)']:.0f}s")

    # save model + confusion matrix picture
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    joblib.dump(pipe, f"{OUT}/models/{safe}.pkl")

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(
        matrices[name], display_labels=["No 30d Readmit", "30d Readmit"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/confusion_matrices/{safe}.png", dpi=120)
    plt.close(fig)

# ---------------------------------------------------------------
# STEP 5: COMPARISON TABLE
# ---------------------------------------------------------------
res = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
res.round(4).to_csv(f"{OUT}/all_models_comparison.csv", index=False)

print("\n" + "=" * 100)
print("ALL MODELS - COMPARISON (sorted by ROC-AUC)")
print("=" * 100)
print(res.round(4).to_string(index=False))

# confusion matrix numbers as text (TN, FP, FN, TP)
print("\nConfusion matrix numbers  [TN  FP  FN  TP]")
for name, cm in matrices.items():
    tn, fp, fn, tp = cm.ravel()
    print(f"  {name:<24} TN={tn:<6} FP={fp:<6} FN={fn:<6} TP={tp}")

# ---------------------------------------------------------------
# STEP 6: PICTURES - bar chart + all confusion matrices in one grid
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
plot_df = res.set_index("Model")[["Test Acc", "Precision", "Recall", "F1", "ROC-AUC"]]
plot_df.plot(kind="bar", ax=ax)
ax.set_title("All models - metric comparison")
ax.set_ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/all_models_bar_chart.png", dpi=130)
plt.close(fig)

n = len(matrices)
cols = 4
rows_n = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4 * rows_n))
axes = np.array(axes).reshape(-1)
for ax, (name, cm) in zip(axes, matrices.items()):
    ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name, fontsize=10)
for ax in axes[n:]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/all_confusion_matrices.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------
# STEP 7: SAVE THE BEST MODEL (by ROC-AUC) as best_model.pkl
# ---------------------------------------------------------------
best_name = res.loc[0, "Model"]
joblib.dump(trained[best_name], f"{OUT}/best_model.pkl")
with open(f"{OUT}/best_model_name.txt", "w") as f:
    f.write(best_name)
print(f"\nBest model by ROC-AUC: {best_name}  -> saved as best_model.pkl")

with open(f"{OUT}/feature_columns.json", "w") as f:
    json.dump({
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "categorical_options": {c: sorted(X[c].dropna().unique().tolist())
                                for c in categorical_cols},
    }, f, indent=2)

print(f"\nAll done. Everything is in the '{OUT}' folder.")
