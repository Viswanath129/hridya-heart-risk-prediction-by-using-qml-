import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import load_and_preprocess_data

# The dataset has 6 features: Age, Gender, BloodPressure, Cholesterol, HeartRate, QuantumPatternFeature
n_qubits = 6
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    # Angle Encoding for classical data
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # Variational layers for non-linear interactions
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Measure expectation value of PauliZ on the first qubit
    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, bias, inputs):
    return qnode(inputs, weights) + bias

def square_loss(labels, predictions):
    # Uses Pennylane numpy to keep gradients
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    return loss / len(labels)

def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)

def evaluate_model(predictions, y_test):
    # Map predictions back to 1 or -1
    y_pred = [1 if p >= 0 else -1 for p in predictions]
    acc = accuracy_score(y_test, y_pred)
    # y_test and y_pred are -1 or 1, we map them to 0 and 1 for metrics
    y_test_bin = [1 if y == 1 else 0 for y in y_test]
    y_pred_bin = [1 if p == 1 else 0 for p in y_pred]
    
    prec = precision_score(y_test_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_test_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
    return acc, prec, rec, f1

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/heart_dataset.csv')
    
    # Map binary class labels (0, 1) to (-1.0, 1.0)
    y_train_q = np.where(y_train == 0, -1.0, 1.0)
    y_test_q = np.where(y_test == 0, -1.0, 1.0)
    
    # Initialize weights
    np.random.seed(42)
    n_layers = 2
    # StronglyEntanglingLayers needs weight shape (n_layers, n_qubits, 3)
    weights = np.random.randn(n_layers, n_qubits, 3, requires_grad=True)
    bias = np.array(0.0, requires_grad=True)
    
    opt = qml.AdamOptimizer(stepsize=0.1)
    batch_size = 16
    epochs = 15
    
    print("Training HRIDYA-QML Validational Quantum Circuit...")
    for it in range(epochs):
        # Update weights step
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = X_train[batch_index]
        Y_batch = y_train_q[batch_index]
        
        weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=Y_batch)
        
        # Compute accuracy on batch
        res = [variational_classifier(weights, bias, x) for x in X_batch]
        acc, _, _, _ = evaluate_model(res, Y_batch)
        print(f"Iter: {it+1:3d} | Cost: {cost(weights, bias, X_batch, Y_batch):.4f} | Batch Acc: {acc:.4f}")
        
    print("Evaluating HRIDYA-QML on test set...")
    predictions_test = [variational_classifier(weights, bias, x) for x in X_test]
    acc, prec, rec, f1 = evaluate_model(predictions_test, y_test_q)
    
    results = [{
        'Model': 'HRIDYA-QML',
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1
    }]
    
    os.makedirs('results/tables', exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/tables/qml_results.csv', index=False)
    print("HRIDYA-QML results saved to results/tables/qml_results.csv")

if __name__ == "__main__":
    main()
