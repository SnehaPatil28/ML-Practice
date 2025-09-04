## Loan Approval

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------
# 1. Load Dataset
# -------------------------
# Example Loan Approval dataset (synthetic for illustration)
data = {
    'Age': [25, 45, 35, 50, 23, 40, 60, 48, 33, 29],
    'Income': [50000, 80000, 60000, 90000, 35000, 70000, 120000, 95000, 58000, 42000],
    'Credit_History': ['Good', 'Good', 'Bad', 'Good', 'Bad', 'Good', 'Good', 'Bad', 'Good', 'Bad'],
    'Loan_Amount': [200, 300, 250, 400, 150, 350, 500, 300, 220, 180],
    'Approved': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# -------------------------
# 2. Data Preprocessing
# -------------------------
# Convert categorical variables to numeric
df['Credit_History'] = df['Credit_History'].map({'Good': 1, 'Bad': 0})
df['Approved'] = df['Approved'].map({'Yes': 1, 'No': 0})

# Features & Target
X = df.drop('Approved', axis=1)
y = df['Approved']

# -------------------------
# 3. Split Data
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------
# 4. Train Decision Tree (Without Pruning)
# -------------------------
clf_no_pruning = DecisionTreeClassifier(random_state=42)
clf_no_pruning.fit(X_train, y_train)

# -------------------------
# 5. Visualize Tree Structure
# -------------------------
plt.figure(figsize=(10,2))
plot_tree(clf_no_pruning, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree Without Pruning")
plt.show()

# -------------------------
# 6. Train Decision Tree (With Pruning)
# -------------------------
clf_pruned = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)
clf_pruned.fit(X_train, y_train)

plt.figure(figsize=(10,2))
plot_tree(clf_pruned, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree With Pruning")
plt.show()

# -------------------------
# 7. Compare Performance
# -------------------------
y_pred_no_pruning = clf_no_pruning.predict(X_test)
y_pred_pruned = clf_pruned.predict(X_test)

acc_no_pruning = accuracy_score(y_test, y_pred_no_pruning)
acc_pruned = accuracy_score(y_test, y_pred_pruned)

print("\nAccuracy without pruning:", acc_no_pruning)
print("Accuracy with pruning:", acc_pruned)

# -------------------------
# 8. Analyze Feature Importance
# -------------------------
print("\nFeature Importances (Pruned Tree):")
for feature, importance in zip(X.columns, clf_pruned.feature_importances_):
    print(f"{feature}: {importance:.4f}")
