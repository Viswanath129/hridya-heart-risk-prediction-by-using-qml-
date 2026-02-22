import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import load_and_preprocess_data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return (
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred, zero_division=0)
    )

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/heart_dataset.csv')
    
    models = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        acc, prec, rec, f1 = evaluate_model(model, X_test, y_test)
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        })
    
    os.makedirs('results/tables', exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/tables/classical_results.csv', index=False)
    print("Classical baselines evaluated and saved to results/tables/classical_results.csv")

if __name__ == "__main__":
    main()
