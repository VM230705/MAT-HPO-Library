"""
MAT-HPO Library - Flexible Evaluation Criteria System

This module provides flexible evaluation criteria for model saving and optimization,
specifically designed to support SPNV2's val_f1_macro primary objective.
"""

from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass
from enum import Enum
import json

class OptimizationTarget(Enum):
    """優化目標枚舉"""
    VAL_F1_MACRO = "val_f1_macro"
    VAL_ACCURACY = "val_accuracy" 
    TEST_F1_MACRO = "test_f1_macro"
    TEST_ACCURACY = "test_accuracy"
    CUSTOM_REWARD = "custom_reward"

@dataclass
class ModelSaveCriteria:
    """模型保存條件配置"""
    # 主要優化目標
    primary_target: OptimizationTarget
    
    # 次要條件（可選）
    secondary_targets: Optional[List[OptimizationTarget]] = None
    
    # 自訂評估函數
    custom_evaluator: Optional[Callable[[Dict[str, Any]], float]] = None
    
    # 保存條件
    save_on_improvement: bool = True
    save_on_milestone: bool = True
    milestone_steps: List[int] = None
    
    # 閾值條件
    min_improvement_threshold: float = 0.0
    absolute_threshold: Optional[float] = None
    
    def __post_init__(self):
        if self.milestone_steps is None:
            self.milestone_steps = [25, 50, 75, 100, 125, 150, 175, 200]

class FlexibleEvaluator:
    """靈活的評估器 - 修復版本"""
    
    def __init__(self, criteria: ModelSaveCriteria):
        self.criteria = criteria
        self.best_score = float('-inf')
        self.best_step = -1  # ✅ 修復：正確追蹤 best_step
        self.best_hyperparams = None
        self.best_metrics = {}
        
    def evaluate(self, metrics: Dict[str, Any], step: int, hyperparams: Dict[str, Any]) -> bool:
        """
        評估是否應該保存模型
        
        Returns:
            bool: 是否保存模型
        """
        # 計算主要分數
        primary_score = self._get_primary_score(metrics)
        
        # 檢查是否改進
        improved = primary_score > self.best_score + self.criteria.min_improvement_threshold
        
        # 檢查絕對閾值
        threshold_met = (self.criteria.absolute_threshold is None or 
                        primary_score >= self.criteria.absolute_threshold)
        
        # 檢查里程碑
        milestone_met = step in self.criteria.milestone_steps
        
        should_save = False
        
        if improved and threshold_met:
            # ✅ 修復：只有在真正改進時才更新 best_step
            self.best_score = primary_score
            self.best_step = step  # 正確記錄發現最佳模型的 step
            self.best_hyperparams = hyperparams.copy()
            self.best_metrics = metrics.copy()
            should_save = True
            
        elif milestone_met and self.criteria.save_on_milestone:
            should_save = True
            
        return should_save
    
    def _get_primary_score(self, metrics: Dict[str, Any]) -> float:
        """獲取主要評估分數"""
        if self.criteria.custom_evaluator:
            return self.criteria.custom_evaluator(metrics)
        
        target_key = self.criteria.primary_target.value
        return metrics.get(target_key, 0.0)
    
    def get_best_info(self) -> Dict[str, Any]:
        """獲取最佳模型信息"""
        return {
            'best_score': self.best_score,
            'best_step': self.best_step,
            'best_hyperparams': self.best_hyperparams,
            'best_metrics': self.best_metrics,
            'primary_target': self.criteria.primary_target.value
        }

def create_spnv2_criteria() -> ModelSaveCriteria:
    """創建 SPNV2 專用的評估標準"""
    return ModelSaveCriteria(
        primary_target=OptimizationTarget.VAL_F1_MACRO,
        secondary_targets=[OptimizationTarget.TEST_F1_MACRO],
        save_on_improvement=True,
        save_on_milestone=True,
        milestone_steps=[25, 50, 75, 100, 125, 150, 175, 200],
        min_improvement_threshold=0.001,  # 最小改進閾值
        absolute_threshold=None  # 無絕對閾值
    )

def create_spnv2_custom_evaluator():
    """創建 SPNV2 自訂評估函數"""
    def custom_evaluator(metrics: Dict[str, Any]) -> float:
        """
        自訂評估函數：主要基於 val_f1_macro，次要考慮 test_f1_macro
        """
        val_f1 = metrics.get('val_f1', 0.0)
        test_f1 = metrics.get('test_f1', 0.0)
        
        # 主要權重給 val_f1，次要權重給 test_f1
        combined_score = val_f1 * 0.8 + test_f1 * 0.2
        
        return combined_score
    
    return custom_evaluator
