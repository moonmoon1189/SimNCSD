import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plot_and_save_results(y_true, y_prob, threshold, output_dir="outputs/figures"):
    """
    Generate and save ROC curve, Confusion Matrix, and Probability Distribution.
    """
    os.makedirs(output_dir, exist_ok=True)
    y_pred = (np.array(y_prob) >= threshold).astype(int)

    # Set plot style
    sns.set_theme(style="whitegrid")

    # 1. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()

    # 2. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Source-Consistent (0)', 'Source-Deviant (1)'],
                yticklabels=['Source-Consistent (0)', 'Source-Deviant (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # 3. Plot Probability Density Distribution
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=[p for t, p in zip(y_true, y_prob) if t == 1], fill=True, color="red", label="Source-Deviant (1)")
    sns.kdeplot(data=[p for t, p in zip(y_true, y_prob) if t == 0], fill=True, color="blue",
                label="Source-Consistent (0)")
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.2f})')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'prob_distribution.png'), dpi=300)
    plt.close()

    print(f"Visualizations successfully saved to: {output_dir}")