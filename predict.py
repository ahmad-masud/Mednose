import numpy as np


def predict_disease(clf, mlb, input_symptoms, top_k=3):
    """
    Predict the most probable diseases based on a list of input symptoms.

    Parameters:
    clf (RandomForestClassifier): Trained classifier used for prediction.
    mlb (MultiLabelBinarizer): Binarizer used to transform symptoms into feature vectors.
    input_symptoms (list of str): List of symptoms provided by the user.
    top_k (int): Number of top disease predictions to return. Default is 3.

    Returns:
    top_predictions (list of tuples): List of (disease, probability) tuples for top predictions.
    unknown (list of str): Symptoms that were not recognized in the model's vocabulary.
    """
    valid_set = set(mlb.classes_)
    filtered = [s for s in input_symptoms if s in valid_set]
    unknown = [s for s in input_symptoms if s not in valid_set]

    if not filtered:
        return [], unknown

    input_vector = mlb.transform([filtered])
    probs = clf.predict_proba(input_vector)[0]
    indices = probs.argsort()[::-1][:top_k]

    top_predictions = [(clf.classes_[idx], probs[idx]) for idx in indices]
    return top_predictions, unknown
