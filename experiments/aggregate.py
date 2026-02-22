import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    classical_df = pd.read_csv('results/tables/classical_results.csv')
    qml_df = pd.read_csv('results/tables/qml_results.csv')
    
    # Aggregate
    summary_df = pd.concat([classical_df, qml_df], ignore_index=True)
    os.makedirs('results/tables', exist_ok=True)
    summary_df.to_csv('results/tables/summary.csv', index=False)
    print("Aggregated results saved to results/tables/summary.csv")
    print("\n--- Summary ---")
    print(summary_df.to_string(index=False))
    
    # Plotting
    os.makedirs('results/plots', exist_ok=True)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('HRIDYA: Classical vs Quantum-Inspired Models Performance')
    
    colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#d62728'] # Highlight QML
    for ax, metric in zip(axes.flatten(), metrics):
        bars = ax.bar(summary_df['Model'], summary_df[metric], color=colors)
        ax.set_title(metric)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        # Add labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('results/plots/comparison_metrics.png', dpi=300)
    print("\nComparison plot saved to results/plots/comparison_metrics.png")

if __name__ == "__main__":
    main()
