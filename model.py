from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def train_model(X, y, n_estimators=100):
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
    importances = clf.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    features = [mlb.classes_[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices], align='center')
    plt.yticks(range(top_n), features)
    plt.xlabel("Importance")
    plt.title("Top Important Symptoms")
    plt.tight_layout()
    plt.show()
