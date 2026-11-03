import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_task_2():
    
    print("***LOGISTIC REGRESSION AND PROBABILITIES***")

    cancer = load_breast_cancer()
    X_cancer, y_cancer = cancer.data, cancer.target
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cancer, y_cancer, test_size=0.2, random_state=42
    )

    binary_model = LogisticRegression(max_iter=10000)
    binary_model.fit(X_train_c, y_train_c)

    binary_acc = accuracy_score(y_test_c, binary_model.predict(X_test_c))
    print(f"No:1 Breast Cancer Accuracy: {binary_acc:.4f}\n")

    probs_5 = binary_model.predict_proba(X_test_c)[:5]
    preds_5 = binary_model.predict(X_test_c)[:5]

    print("No:2 First 5 Test Rows Comparison (Default Threshold = 0.5):")
    for i in range(5):
        print(f"  Row {i+1} -> Probabilities [Class 0, Class 1]: {probs_5[i].round(4)} | Predicted Class: {preds_5[i]}")
    
    positive_class_probs = binary_model.predict_proba(X_test_c)[:, 1]
    preds_threshold_03 = (positive_class_probs >= 0.3).astype(int)

    print("No:3 Comparison of Predictions with Lower Threshold (0.3):")
    for i in range(5):
        print(f"  Row {i+1} -> P(Class 1): {positive_class_probs[i]:.4f} | Default (0.5): {preds_5[i]} | New (0.3): {preds_threshold_03[i]}")

    
    iris = load_iris()
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    multi_model = LogisticRegression(max_iter=10000)
    multi_model.fit(X_train_i, y_train_i)

    iris_probs = multi_model.predict_proba(X_test_i)[:5]
    iris_preds = multi_model.predict(X_test_i)[:5]

    iris_df = pd.DataFrame(iris_probs, columns=iris.target_names)
    iris_df['predicted_class'] = [iris.target_names[p] for p in iris_preds]
    
    print("No:4 MULTI CLASS PROBABILITY TABLE FOR IRIS:")
    print(iris_df.to_string(index=False))

    print("No:5 Reason Why Probabilites are useful:")
    print("1. Probabilities quantify the model's confidence level rather than forcing a blind binary guess.")
    print("2. They allow us to flexibly adjust decision thresholds based on real-world risk vs reward scenarios.")
    print("3. Soft prediction scales capture subtle edge cases, improving downstream metric evaluations like ROC-AUC.")

if __name__ == "__main__":
    run_task_2()