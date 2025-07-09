# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Load the Wine Quality dataset
df = pd.read_csv("winequality-red.csv")

print(df.head())

# Features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Optionally, make this a binary classification (good wine vs bad wine)
# Many people do this:
y = y.apply(lambda x: 1 if x >= 7 else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Models
models = {
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

# Hyperparameters
param_grid = {
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10]
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    },
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    }
}

# Results storage
best_models = {}
results = []

# Train and tune
for name, model in models.items():
    print(f"\nTraining & tuning {name}...")

    grid = GridSearchCV(model, param_grid[name], cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    print(f"Best Params: {grid.best_params_}")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")

    best_models[name] = grid.best_estimator_
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1
    })

# Compare models
results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:\n")
print(results_df)

# Best model
best_model = results_df.sort_values(by="F1-score", ascending=False).iloc[0]
print(f"\nBest model: {best_model['Model']} with F1-score: {best_model['F1-score']:.4f}")
