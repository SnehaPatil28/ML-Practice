## Iris Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()
X= pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_no_pruning = DecisionTreeClassifier(random_state=42)
clf_no_pruning.fit(X_train, y_train)

plt.figure(figsize=(12, 4))
plot_tree(
    clf_no_pruning,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
)

plt.title("Decision Tree without Pruning")
plt.show()

clf_pruned = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2 ,random_state=42)
clf_pruned.fit(X_train, y_train)

plt.figure(figsize=(12, 4))
plot_tree(
    clf_pruned,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
)

plt.title("Desicion Tree with Pruning")
plt.show()

y_pred_no_pruning = clf_no_pruning.predict(X_test)
y_pred_pruned = clf_pruned.predict(X_test)

acc_no_pruning = accuracy_score(y_test, y_pred_no_pruning)
acc_pruned = accuracy_score(y_test, y_pred_pruned)

print("\nAccuracy without pruning:", acc_no_pruning)
print("Accuracy with pruning:", acc_pruned)

for feature, importance in zip(iris.feature_names, clf_pruned.feature_importances_):
    print(f"{feature}: {importance:.4f}")
