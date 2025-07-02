"""
Evaluation metrics for HAR models.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Model evaluation utilities for HAR.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize the evaluator.
        
        Args:
            class_names: List of activity class names.
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Evaluate model predictions.
        
        Args:
            y_true: True labels (one-hot encoded or integer).
            y_pred: Predicted labels (one-hot encoded or integer).
            y_pred_proba: Prediction probabilities.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        # Convert one-hot to integers if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Per-class metrics
        class_report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        metrics['classification_report'] = class_report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Per-class accuracy
        per_class_accuracy = []
        for i in range(self.num_classes):
            class_mask = y_true == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
                per_class_accuracy.append(class_acc)
            else:
                per_class_accuracy.append(0.0)
        metrics['per_class_accuracy'] = per_class_accuracy
        
        # If probabilities are provided, calculate additional metrics
        if y_pred_proba is not None:
            metrics['precision_recall_curves'] = self._calculate_pr_curves(y_true, y_pred_proba)
            metrics['roc_curves'] = self._calculate_roc_curves(y_true, y_pred_proba)
        
        return metrics
    
    def _calculate_pr_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate precision-recall curves for each class.
        
        Args:
            y_true: True labels (integers).
            y_pred_proba: Prediction probabilities.
            
        Returns:
            Dictionary with PR curve data for each class.
        """
        pr_curves = {}
        
        for i, class_name in enumerate(self.class_names):
            # Binarize for current class
            y_true_binary = (y_true == i).astype(int)
            y_scores = y_pred_proba[:, i]
            
            # Calculate PR curve
            precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores)
            
            pr_curves[class_name] = {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds,
                'auc': auc(recall, precision)
            }
        
        return pr_curves
    
    def _calculate_roc_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate ROC curves for each class.
        
        Args:
            y_true: True labels (integers).
            y_pred_proba: Prediction probabilities.
            
        Returns:
            Dictionary with ROC curve data for each class.
        """
        roc_curves = {}
        
        for i, class_name in enumerate(self.class_names):
            # Binarize for current class
            y_true_binary = (y_true == i).astype(int)
            y_scores = y_pred_proba[:, i]
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
            
            roc_curves[class_name] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': auc(fpr, tpr)
            }
        
        return roc_curves
    
    def print_evaluation_summary(self, metrics: Dict[str, any]):
        """
        Print a summary of evaluation metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics.
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION SUMMARY")
        print("="*50)
        
        # Overall metrics
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-averaged Precision: {metrics['precision_macro']:.4f}")
        print(f"Macro-averaged Recall: {metrics['recall_macro']:.4f}")
        print(f"Macro-averaged F1-score: {metrics['f1_macro']:.4f}")
        
        # Per-class metrics
        print("\nPer-Class Performance:")
        print("-"*50)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Accuracy':<10}")
        print("-"*50)
        
        for i, class_name in enumerate(self.class_names):
            class_metrics = metrics['classification_report'][class_name]
            class_acc = metrics['per_class_accuracy'][i]
            print(f"{class_name:<15} "
                  f"{class_metrics['precision']:<10.4f} "
                  f"{class_metrics['recall']:<10.4f} "
                  f"{class_metrics['f1-score']:<10.4f} "
                  f"{class_acc:<10.4f}")
        
        print("-"*50)
        
        # Confusion matrix summary
        cm = metrics['confusion_matrix']
        print("\nConfusion Matrix Summary:")
        print(f"Total Correct Predictions: {np.trace(cm)}")
        print(f"Total Misclassifications: {np.sum(cm) - np.trace(cm)}")
        
        # Most confused pairs
        cm_copy = cm.copy()
        np.fill_diagonal(cm_copy, 0)
        if np.max(cm_copy) > 0:
            max_confused = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
            print(f"Most Confused Pair: {self.class_names[max_confused[0]]} → "
                  f"{self.class_names[max_confused[1]]} ({cm_copy[max_confused]} samples)")
    
    def save_metrics(self, metrics: Dict[str, any], save_path: str):
        """
        Save evaluation metrics to file.
        
        Args:
            metrics: Dictionary of evaluation metrics.
            save_path: Path to save the metrics.
        """
        import json
        from pathlib import Path
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_metrics[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_metrics[key][k] = v.tolist()
                    else:
                        serializable_metrics[key][k] = v
            else:
                serializable_metrics[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logger.info(f"Metrics saved to: {save_path}")