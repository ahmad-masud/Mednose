from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np


def train_model(X, y, n_estimators=100):
    """
    Train a Random Forest classifier on the given features and labels.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target labels.
    n_estimators (int): Number of trees in the forest. Default is 100.

    Returns:
    clf (RandomForestClassifier): The trained classifier.
    X_test (array-like): Test features.
    y_test (array-like): True labels for the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return clf, X_test, y_test


def plot_feature_importance(clf, mlb, top_n=20):
    """
    Plot the top N most important features from a trained Random Forest model.

    Parameters:
    clf (RandomForestClassifier): Trained classifier with feature importances.
    mlb (MultiLabelBinarizer): Fitted binarizer to map feature indices to names.
    top_n (int): Number of top features to display. Default is 20.

    Returns:
    None
    """
    importances = clf.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    features = [mlb.classes_[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices], align="center")
    plt.yticks(range(top_n), features)
    plt.xlabel("Importance")
    plt.title("Top Important Symptoms")
    plt.tight_layout()
    plt.show()
