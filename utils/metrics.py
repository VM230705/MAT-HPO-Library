"""
Metrics calculation utilities for MAT-HPO
"""

import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import warnings


class MetricsCalculator:
    """
    Utility class for calculating various performance metrics.
    
    Provides standardized metric calculations that can be used across
    different optimization problems.
    """
    
    def __init__(self):
        self.supported_metrics = [
            'accuracy', 'f1', 'precision', 'recall', 'auc', 
            'gmean', 'weighted_f1', 'macro_f1', 'micro_f1'
        ]
    
    def calculate_basic_metrics(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for AUC calculation)
            
        Returns:
            Dictionary containing calculated metrics
        """
        metrics = {}
        
        try:
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Different F1 averaging methods
            metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
            
            # AUC calculation (if probabilities provided)
            if y_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:
                        # Binary classification
                        metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
                    else:
                        # Multi-class classification
                        metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                except ValueError:
                    metrics['auc'] = 0.0
            else:
                metrics['auc'] = metrics['accuracy']  # Fallback to accuracy
            
            # G-mean calculation
            metrics['gmean'] = self.calculate_gmean(y_true, y_pred)
            
        except Exception as e:
            warnings.warn(f"Error calculating metrics: {e}")
            # Return zero metrics if calculation fails
            for metric in self.supported_metrics:
                metrics[metric] = 0.0
        
        return metrics
    
    def calculate_gmean(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate geometric mean of per-class sensitivities.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            G-mean value
        """
        try:
            unique_classes = np.unique(y_true)
            sensitivities = []
            
            for cls in unique_classes:
                # True positives and actual positives for this class
                tp = np.sum((y_true == cls) & (y_pred == cls))
                actual_pos = np.sum(y_true == cls)
                
                if actual_pos > 0:
                    sensitivity = tp / actual_pos
                else:
                    sensitivity = 1.0  # No samples of this class
                
                sensitivities.append(sensitivity)
            
            # Geometric mean
            if sensitivities:
                gmean = np.prod(sensitivities) ** (1.0 / len(sensitivities))
            else:
                gmean = 0.0
            
            return float(gmean)
            
        except Exception:
            return 0.0
    
    def calculate_mauc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Calculate Multi-class Area Under Curve (MAUC).
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities [n_samples, n_classes]
            
        Returns:
            MAUC value
        """
        try:
            num_classes = len(np.unique(y_true))
            if num_classes < 2:
                return 0.0
            
            auc_sum = 0
            count = 0
            
            # Calculate AUC for each pair of classes
            for i in range(num_classes):
                for j in range(i + 1, num_classes):
                    # Create binary problem for classes i and j
                    binary_mask = (y_true == i) | (y_true == j)
                    if np.sum(binary_mask) == 0:
                        continue
                    
                    binary_true = y_true[binary_mask]
                    binary_proba = y_proba[binary_mask, i]
                    
                    # Convert to binary labels (class i = 1, class j = 0)
                    binary_true = (binary_true == i).astype(int)
                    
                    try:
                        auc = roc_auc_score(binary_true, binary_proba)
                        auc_sum += auc
                        count += 1
                    except ValueError:
                        pass  # Skip if AUC cannot be calculated
            
            # Calculate MAUC
            if count > 0:
                mauc = 2 * auc_sum / (num_classes * (num_classes - 1))
            else:
                mauc = 0.0
            
            return float(mauc)
            
        except Exception:
            return 0.0
    
    def calculate_custom_reward(self, 
                               metrics: Dict[str, float],
                               weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate custom reward based on multiple metrics.
        
        Args:
            metrics: Dictionary of metric values
            weights: Dictionary of metric weights (if None, uses equal weights)
            
        Returns:
            Weighted reward value
        """
        if weights is None:
            # Default weights
            weights = {
                'f1': 0.4,
                'accuracy': 0.3,
                'precision': 0.15,
                'recall': 0.15
            }
        
        reward = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                reward += metrics[metric] * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            reward /= total_weight
        
        return float(reward * 100)  # Scale to 0-100 range
    
    def get_metric_summary(self, metrics: Dict[str, float]) -> str:
        """
        Get a formatted summary of metrics.
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            Formatted string summary
        """
        summary = "Metrics Summary:\n"
        summary += "-" * 20 + "\n"
        
        for metric, value in metrics.items():
            summary += f"{metric.capitalize():12}: {value:.4f}\n"
        
        return summary
    
    def compare_metrics(self, 
                       metrics1: Dict[str, float], 
                       metrics2: Dict[str, float]) -> Dict[str, float]:
        """
        Compare two sets of metrics.
        
        Args:
            metrics1: First set of metrics
            metrics2: Second set of metrics
            
        Returns:
            Dictionary of metric differences (metrics2 - metrics1)
        """
        differences = {}
        
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        
        for metric in common_metrics:
            differences[metric] = metrics2[metric] - metrics1[metric]
        
        return differences


# Convenience functions for common calculations
def calculate_f1_auc_gmean(y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          y_proba: Optional[np.ndarray] = None) -> tuple:
    """
    Calculate F1, AUC, and G-mean metrics (commonly used in MAT-HPO).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        y_proba: Predicted probabilities
        
    Returns:
        Tuple of (f1, auc, gmean)
    """
    calculator = MetricsCalculator()
    metrics = calculator.calculate_basic_metrics(y_true, y_pred, y_proba)
    
    return metrics['f1'], metrics['auc'], metrics['gmean']


def simple_reward(f1: float, auc: float, gmean: float, weights: tuple = (0.4, 0.3, 0.3)) -> float:
    """
    Calculate simple weighted reward from F1, AUC, and G-mean.
    
    Args:
        f1: F1 score
        auc: AUC score  
        gmean: G-mean score
        weights: Tuple of weights for (f1, auc, gmean)
        
    Returns:
        Weighted reward value
    """
    return (f1 * weights[0] + auc * weights[1] + gmean * weights[2]) * 100