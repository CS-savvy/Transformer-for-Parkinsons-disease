from sklearn import metrics
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

this_dir = Path.cwd()


def evaluate(model_dir, data, split_details=None):
    conf_threshold = 0.5
    features, labels = data
    max_val_accuracy = 0
    best_model = None
    results = []
    for i in range(1, 11):
        val_indices = split_details[f'val_{i}']
        X_val, y_val = features[val_indices], labels[val_indices]
        model_path = model_dir / f"model_k_fold_{i}.pkl"
        clf = joblib.load(model_path)
        y_score = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_score > conf_threshold)*1.0
        accuracy = metrics.accuracy_score(y_val, y_pred)
        precision = metrics.precision_score(y_val, y_pred)
        recall = metrics.recall_score(y_val, y_pred)
        print(f"Accuracy : {accuracy} | precision : {precision} | Recall : {recall}")
        if accuracy > max_val_accuracy:
            max_val_accuracy = accuracy
            best_model = i
        results.extend([(u, l, p, s, l == p) for u, l, p, s in zip(val_indices, y_val, y_pred, y_score)])

    print(f"Best Model : {best_model} with accuracy {max_val_accuracy}")
    test_indices = split_details['test']
    X_test, y_test = features[test_indices], labels[test_indices]
    model_path = model_dir / f"model_k_fold_{best_model}.pkl"
    clf = joblib.load(model_path)
    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_score > conf_threshold) * 1.0
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    print(f"Test - Accuracy : {accuracy} | precision : {precision} | Recall : {recall}")
    results.extend([(u, l, p, s, l==p) for u, l, p, s in zip(test_indices, y_test, y_pred, y_score)])

    results = sorted(results, key=lambda x: x[0])
    result_df = pd.DataFrame(results, columns=["UID", 'True Label', 'Prediction', 'Score', 'Match'])
    result_df.to_csv(model_dir / f"eval_result.csv", index=False)


if __name__ == "__main__":

    csv_file = this_dir / "data/pd_speech_features.csv"
    model_dir = this_dir / "models/SVM"
    df = pd.read_csv(csv_file, skiprows=[0])
    df.drop(columns=['id'], inplace=True)
    skip_column = ['gender', 'class']
    columns = list(df.columns)
    columns = [c for c in columns if c not in skip_column]
    for col in columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

    data = df.to_numpy(dtype=np.float32)
    features, labels = data[:, :-1], data[:, -1]

    with open("data/split_details.json", 'r', encoding='utf8') as f:
        split_detail = json.load(f)

    evaluate(model_dir, (features, labels), split_details=split_detail)
