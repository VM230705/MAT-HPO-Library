"""
WandB Standards for MAT-HPO Library

This module provides standardized WandB logging formats to ensure consistency
across all MAT-HPO Library components and SPNV2 integration.
"""

from typing import Dict, Any, Optional
import time

class WandBStandards:
    """WandB 記錄標準化"""
    
    @staticmethod
    def get_unified_log_format(step: int, metrics: Dict[str, Any], 
                             hyperparams: Dict[str, Any], 
                             best_metrics: Dict[str, Any],
                             optimization_target: str = "val_f1_macro") -> Dict[str, Any]:
        """
        統一的 WandB 記錄格式
        
        Args:
            step: Current optimization step
            metrics: Current step metrics
            hyperparams: Current step hyperparameters
            best_metrics: Global best metrics so far
            optimization_target: Primary optimization target
            
        Returns:
            Standardized log dictionary
        """
        return {
            # Step 識別
            "step": step,
            "optimization_target": optimization_target,
            
            # 當前 step 的 best epoch 指標（基於 val_f1_macro 選擇）
            "val_f1": metrics.get('val_f1', 0.0),
            "val_acc": metrics.get('val_acc', 0.0),
            "val_precision": metrics.get('val_precision', 0.0),
            "val_recall": metrics.get('val_recall', 0.0),
            
            "test_f1": metrics.get('test_f1', 0.0),
            "test_acc": metrics.get('test_acc', 0.0),
            "test_precision": metrics.get('test_precision', 0.0),
            "test_recall": metrics.get('test_recall', 0.0),
            "test_auc": metrics.get('test_auc', 0.0),
            "test_gmean": metrics.get('test_gmean', 0.0),
            
            # 全局最佳指標（所有 steps 中最好的）
            "best_val_f1": best_metrics.get('val_f1', 0.0),
            "best_val_acc": best_metrics.get('val_acc', 0.0),
            "best_test_f1": best_metrics.get('test_f1', 0.0),
            "best_test_acc": best_metrics.get('test_acc', 0.0),
            
            # 超參數（統一前綴）
            **{f"hyperparam_{k}": v for k, v in hyperparams.items()},
            
            # 時間戳
            "timestamp": time.time()
        }
    
    @staticmethod
    def get_final_summary_format(best_step: int, best_epoch: int,
                               best_metrics: Dict[str, Any],
                               best_hyperparams: Dict[str, Any],
                               optimization_target: str = "val_f1_macro") -> Dict[str, Any]:
        """
        最終摘要格式
        
        Args:
            best_step: Step where best model was found
            best_epoch: Epoch where best model was found
            best_metrics: Best model metrics
            best_hyperparams: Best model hyperparameters
            optimization_target: Primary optimization target
            
        Returns:
            Final summary dictionary
        """
        return {
            # 最佳模型信息
            "best_step": best_step,
            "best_epoch": best_epoch,
            "optimization_target": optimization_target,
            
            # 最終模型指標（基於 best_val_f1 選擇的模型）
            "final_val_f1": best_metrics.get('val_f1', 0.0),
            "final_val_acc": best_metrics.get('val_acc', 0.0),
            "final_val_precision": best_metrics.get('val_precision', 0.0),
            "final_val_recall": best_metrics.get('val_recall', 0.0),
            
            "final_test_f1": best_metrics.get('test_f1', 0.0),
            "final_test_acc": best_metrics.get('test_acc', 0.0),
            "final_test_precision": best_metrics.get('test_precision', 0.0),
            "final_test_recall": best_metrics.get('test_recall', 0.0),
            "final_test_auc": best_metrics.get('test_auc', 0.0),
            "final_test_gmean": best_metrics.get('test_gmean', 0.0),
            
            # 最佳超參數（統一前綴）
            **{f"final_{k}": v for k, v in best_hyperparams.items()},
            
            # 元數據
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "model_selection_criteria": f"best_{optimization_target}"
        }
    
    @staticmethod
    def get_step_log_format(step: int, reward: float, metrics: Dict[str, Any],
                          hyperparams: Dict[str, Any], step_time: float,
                          best_score: float, best_step: int) -> Dict[str, Any]:
        """
        步驟日誌格式（用於 step_log.jsonl）
        
        Args:
            step: Current step
            reward: Current step reward
            metrics: Current step metrics
            hyperparams: Current step hyperparameters
            step_time: Time taken for this step
            best_score: Best score so far
            best_step: Step where best score was achieved
            
        Returns:
            Step log dictionary
        """
        return {
            "step": step,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "metrics": {k: float(v) for k, v in metrics.items()},
            "timing": {
                "step_time": step_time,
                "avg_step_time": step_time,  # Will be updated by logger
                "total_time": 0.0,  # Will be updated by logger
                "steps_per_second": 1.0 / step_time if step_time > 0 else 0.0
            },
            "hyperparameters": hyperparams,
            "statistics": {
                "min_step_time": step_time,
                "max_step_time": step_time,
                "step_time_std": 0.0
            },
            "best_tracking": {
                "best_score": best_score,
                "best_step": best_step,
                "steps_since_improvement": step - best_step
            }
        }

class WandBConsistencyChecker:
    """WandB 一致性檢查器"""
    
    @staticmethod
    def validate_metrics_consistency(metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        驗證指標一致性
        
        Args:
            metrics: Metrics dictionary to validate
            
        Returns:
            Dictionary of validation results
        """
        issues = {}
        
        # 檢查必要的指標
        required_metrics = ['val_f1', 'val_acc', 'test_f1', 'test_acc']
        for metric in required_metrics:
            if metric not in metrics:
                issues[metric] = f"Missing required metric: {metric}"
            elif not isinstance(metrics[metric], (int, float)):
                issues[metric] = f"Invalid type for {metric}: {type(metrics[metric])}"
        
        # 檢查指標範圍
        for metric in ['val_f1', 'test_f1']:
            if metric in metrics:
                value = metrics[metric]
                if not (0.0 <= value <= 1.0):
                    issues[metric] = f"Value out of range [0,1]: {value}"
        
        # 檢查一致性
        if 'val_f1' in metrics and 'test_f1' in metrics:
            val_f1 = metrics['val_f1']
            test_f1 = metrics['test_f1']
            if abs(val_f1 - test_f1) > 0.5:  # 如果差異過大
                issues['f1_consistency'] = f"Large F1 difference: val={val_f1:.3f}, test={test_f1:.3f}"
        
        return issues
    
    @staticmethod
    def validate_hyperparams_consistency(hyperparams: Dict[str, Any]) -> Dict[str, str]:
        """
        驗證超參數一致性
        
        Args:
            hyperparams: Hyperparameters dictionary to validate
            
        Returns:
            Dictionary of validation results
        """
        issues = {}
        
        # 檢查必要的超參數
        required_params = ['learning_rate', 'batch_size', 'hidden_size']
        for param in required_params:
            if param not in hyperparams:
                issues[param] = f"Missing required hyperparameter: {param}"
        
        # 檢查學習率範圍
        if 'learning_rate' in hyperparams:
            lr = hyperparams['learning_rate']
            if not (1e-6 <= lr <= 1.0):
                issues['learning_rate'] = f"Learning rate out of reasonable range: {lr}"
        
        # 檢查批次大小
        if 'batch_size' in hyperparams:
            bs = hyperparams['batch_size']
            if not (1 <= bs <= 1024):
                issues['batch_size'] = f"Batch size out of reasonable range: {bs}"
        
        return issues
