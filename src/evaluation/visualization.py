"""
Visualization utilities for HAR model evaluation.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """
    Visualization utilities for HAR models.
    """
    
    def __init__(self, class_names: List[str], save_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            class_names: List of activity class names.
            save_dir: Directory to save plots. If None, plots are shown but not saved.
        """
        self.class_names = class_names
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title_suffix: str = '',
        save_name: Optional[str] = None
    ):
        """
        Plot training history (accuracy and loss).
        
        Args:
            history: Training history dictionary.
            title_suffix: Suffix for plot titles.
            save_name: Name for saving the plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(history['accuracy'], label='Train Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'Model Accuracy{title_suffix}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='lower right')
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history['loss'], label='Train Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title(f'Model Loss{title_suffix}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper right')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        title: str = 'Confusion Matrix',
        save_name: Optional[str] = None,
        normalize: bool = False,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix.
            title: Plot title.
            save_name: Name for saving the plot.
            normalize: Whether to normalize the confusion matrix.
            figsize: Figure size.
        """
        plt.figure(figsize=figsize)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            square=True,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(
        self,
        pr_curves: Dict[str, Dict[str, np.ndarray]],
        title: str = 'Precision-Recall Curves',
        save_name: Optional[str] = None
    ):
        """
        Plot precision-recall curves for all classes.
        
        Args:
            pr_curves: Dictionary with PR curve data for each class.
            title: Plot title.
            save_name: Name for saving the plot.
        """
        plt.figure(figsize=(10, 8))
        
        for class_name, pr_data in pr_curves.items():
            plt.plot(
                pr_data['recall'],
                pr_data['precision'],
                label=f"{class_name} (AUC = {pr_data['auc']:.3f})"
            )
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(
        self,
        roc_curves: Dict[str, Dict[str, np.ndarray]],
        title: str = 'ROC Curves',
        save_name: Optional[str] = None
    ):
        """
        Plot ROC curves for all classes.
        
        Args:
            roc_curves: Dictionary with ROC curve data for each class.
            title: Plot title.
            save_name: Name for saving the plot.
        """
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for class_name, roc_data in roc_curves.items():
            plt.plot(
                roc_data['fpr'],
                roc_data['tpr'],
                label=f"{class_name} (AUC = {roc_data['auc']:.3f})"
            )
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_class_metrics(
        self,
        metrics: Dict[str, any],
        save_name: Optional[str] = None
    ):
        """
        Plot per-class performance metrics.
        
        Args:
            metrics: Dictionary with evaluation metrics.
            save_name: Name for saving the plot.
        """
        # Extract per-class metrics
        class_report = metrics['classification_report']
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for class_name in self.class_names:
            if class_name in class_report:
                precision_scores.append(class_report[class_name]['precision'])
                recall_scores.append(class_report[class_name]['recall'])
                f1_scores.append(class_report[class_name]['f1-score'])
        
        # Create bar plot
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision_scores, width, label='Precision')
        bars2 = ax.bar(x, recall_scores, width, label='Recall')
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-score')
        
        ax.set_xlabel('Activity Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_predictions(
        self,
        X_samples: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_samples: int = 5,
        save_name: Optional[str] = None
    ):
        """
        Plot sample predictions with accelerometer data.
        
        Args:
            X_samples: Sample sequences.
            y_true: True labels.
            y_pred: Predicted labels.
            num_samples: Number of samples to plot.
            save_name: Name for saving the plot.
        """
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
        
        if num_samples == 1:
            axes = [axes]
        
        # Select random samples
        indices = np.random.choice(len(X_samples), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            
            # Plot accelerometer data
            time_steps = np.arange(X_samples[idx].shape[0])
            ax.plot(time_steps, X_samples[idx][:, 0], label='X-axis', alpha=0.8)
            ax.plot(time_steps, X_samples[idx][:, 1], label='Y-axis', alpha=0.8)
            ax.plot(time_steps, X_samples[idx][:, 2], label='Z-axis', alpha=0.8)
            
            # Add labels
            true_label = self.class_names[y_true[idx]]
            pred_label = self.class_names[y_pred[idx]]
            
            title = f'True: {true_label}, Predicted: {pred_label}'
            if true_label != pred_label:
                title += ' (INCORRECT)'
                ax.set_facecolor('#ffeeee')
            
            ax.set_title(title)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Acceleration (normalized)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_evaluation_report(
        self,
        metrics: Dict[str, any],
        history: Optional[Dict[str, List[float]]] = None,
        model_name: str = 'Model'
    ):
        """
        Create a comprehensive evaluation report with multiple visualizations.
        
        Args:
            metrics: Dictionary with evaluation metrics.
            history: Training history dictionary.
            model_name: Name of the model for report titles.
        """
        logger.info(f"Creating evaluation report for {model_name}")
        
        # Plot training history if available
        if history:
            self.plot_training_history(
                history,
                title_suffix=f' - {model_name}',
                save_name=f'{model_name}_training_history'
            )
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            metrics['confusion_matrix'],
            title=f'Confusion Matrix - {model_name}',
            save_name=f'{model_name}_confusion_matrix'
        )
        
        # Plot normalized confusion matrix
        self.plot_confusion_matrix(
            metrics['confusion_matrix'],
            title=f'Normalized Confusion Matrix - {model_name}',
            save_name=f'{model_name}_confusion_matrix_normalized',
            normalize=True
        )
        
        # Plot per-class metrics
        self.plot_per_class_metrics(
            metrics,
            save_name=f'{model_name}_per_class_metrics'
        )
        
        # Plot PR curves if available
        if 'precision_recall_curves' in metrics:
            self.plot_precision_recall_curves(
                metrics['precision_recall_curves'],
                title=f'Precision-Recall Curves - {model_name}',
                save_name=f'{model_name}_pr_curves'
            )
        
        # Plot ROC curves if available
        if 'roc_curves' in metrics:
            self.plot_roc_curves(
                metrics['roc_curves'],
                title=f'ROC Curves - {model_name}',
                save_name=f'{model_name}_roc_curves'
            )
        
        logger.info("Evaluation report completed")
