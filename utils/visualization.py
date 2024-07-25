import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np

def plot_loss_accuracy(history, output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path, dpi=800)
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    accuracy_plot_path = os.path.join(output_dir, 'accuracy_plot.png')
    plt.savefig(accuracy_plot_path, dpi=800)
    plt.close()

    print(f"Loss and accuracy plots saved to {output_dir}")

def plot_confusion_matrix(y_true, y_pred, output_dir):
     """
    Plot the confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.grid(False)
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=800)
    plt.close()

    print(f"Confusion matrix plot saved to {output_dir}")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
def plot_model_performance(performance_values, output_dir):
    """
    Plot the model performance metrics.

    Args:
        performance_values: List of performance metrics [accuracy, loss, precision, recall, f1].
        output_dir: Directory to save the plot.
        save_path: Path to save the performance plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # Labels for the performance metrics
    labels = ['Accuracy', 'Loss', 'Precision', 'Recall', 'F1 Score']
    
    # Plot bar chart
    ax.bar(labels, performance_values, color='skyblue')

    # Add data labels
    for i in range(len(labels)):
        ax.text(i, performance_values[i], '{:.4f}'.format(performance_values[i]), ha='center', va='bottom', color='black')

    # Set titles and labels
    ax.set_ylabel('Performance')
    ax.set_title('Model Performance')
    
    # Save the plot
    plt.savefig(outpout_dir, format='png', bbox_inches='tight', dpi=800)
    plt.close()

def plot_roc_curve(y_test, y_pred, output_dir):
    """
    Plot the ROC curve.

    Args:
        y_true: True labels.
        y_pred: Predicted probabilities.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    roc_curve_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_curve_path)
    plt.close()

    print(f"ROC curve plot saved to {output_dir}")

