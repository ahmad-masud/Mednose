import numpy as np

def predict_disease(clf, mlb, input_symptoms, top_k=3):
    valid_set = set(mlb.classes_)
    filtered = [s for s in input_symptoms if s in valid_set]
    unknown = [s for s in input_symptoms if s not in valid_set]

    if unknown:
        print(f"Unknown symptoms ignored: {', '.join(unknown)}")

    input_vector = mlb.transform([filtered])
    probs = clf.predict_proba(input_vector)[0]
    indices = probs.argsort()[::-1][:top_k]

    for idx in indices:
        print(f"{clf.classes_[idx]} (Confidence: {probs[idx] * 100:.1f}%)")
