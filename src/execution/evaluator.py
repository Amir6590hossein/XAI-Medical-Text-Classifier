import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

class Evaluator:
    @staticmethod
    def get_uncertainty_statement(entropy):
        if entropy < 0.6:
            return "Low Uncertainty (Confident)"
        elif entropy < 1.1:
            return "Moderate Uncertainty"
        else:
            return "High Uncertainty (Caution Needed)"

    @staticmethod
    def calculate_ece(probs, labels, n_bins=10):
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)
        confidences = confidences.cpu().numpy()
        accuracies = accuracies.cpu().numpy()
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    @staticmethod
    def plot_calibration_curve(probs, labels, save_path, n_bins=10):
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)
        conf_np = confidences.cpu().numpy()
        acc_np = accuracies.cpu().numpy().astype(int)
        
        prob_true, prob_pred = calibration_curve(acc_np, conf_np, n_bins=n_bins, strategy='uniform')
        ece_score = Evaluator.calculate_ece(probs, labels, n_bins)

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated', color='gray')
        plt.plot(prob_pred, prob_true, marker='o', label=f'Model (ECE = {ece_score:.4f})', color='blue')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()
        return ece_score

    @staticmethod
    def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_training_history(history, save_path):
        epochs = range(1, len(history['train_loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', color='tab:blue')
        plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o', color='tab:red')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o', color='tab:blue')
        plt.plot(epochs, history['val_acc'], label='Val Acc', marker='o', color='tab:green')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_roc_curve(probs, targets, n_classes, save_path):
        targets_bin = label_binarize(targets.cpu().numpy(), classes=range(n_classes))
        probs_np = probs.cpu().numpy()
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plt.figure(figsize=(8, 6))
        colors = plt.cm.get_cmap('tab10', n_classes)

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(targets_bin[:, i], probs_np[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
                     label=f'Grade {i+1} (AUC = {roc_auc[i]:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_data_distribution(labels, class_names, save_path, title="Class Distribution"):
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
            
        unique, counts = np.unique(labels, return_counts=True)
        count_dict = dict(zip(unique, counts))
        final_counts = [count_dict.get(i, 0) for i in range(len(class_names))]
        
        plt.figure(figsize=(8, 5))
        bars = plt.bar(class_names, final_counts, color='skyblue', edgecolor='black')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontweight='bold')
            
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.title(title)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def compute(outputs, targets, group_targets):
        probs = F.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
        avg_entropy = entropy.mean().item()
        uncertainty_msg = Evaluator.get_uncertainty_statement(avg_entropy)

        _, preds = torch.max(outputs, 1)
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        acc_grade = accuracy_score(targets_np, preds_np)
        f1_grade = f1_score(targets_np, preds_np, average='macro')
        ece_score = Evaluator.calculate_ece(probs, targets, n_bins=10)
        
        return {
            'accuracy': acc_grade,
            'f1_macro': f1_grade,
            'entropy': avg_entropy,
            'uncertainty_statement': uncertainty_msg,
            'ece': ece_score,
            'conf_matrix': confusion_matrix(targets_np, preds_np).tolist(),
            'group_accuracy': 0.0 # Placeholder
        }