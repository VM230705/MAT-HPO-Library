#!/usr/bin/env python3
"""
時間序列預測的MAT-HPO通用接口示例

展示如何使用MAT-HPO-Library的通用接口來：
1. 自訂metrics（如MASE, SMAPE, MAE, RMSE）
2. 自訂reward function
3. 自訂logger的metrics提取器
"""

import sys
sys.path.append('/home/vm230705/research/MAT-HPO-Library')
sys.path.append('/home/vm230705/research/nnts')

from MAT_HPO_LIB import MAT_HPO_Optimizer, BaseEnvironment, HyperparameterSpace
from MAT_HPO_LIB.utils import DefaultConfigs
from MAT_HPO_LIB.utils.logger import HPOLogger
import numpy as np

# ============================================================================
# 步驟 1: 定義自訂的metrics提取器
# ============================================================================

def timeseries_metrics_extractor(hyperparams: dict) -> dict:
    """
    從hyperparams中提取所有時間序列相關的原始指標
    
    這個函數定義了要記錄哪些指標，以及如何從hyperparams中提取它們
    
    Args:
        hyperparams: 包含所有參數和指標的字典
        
    Returns:
        包含所有要記錄的原始指標的字典
    """
    metrics = {}
    
    # 核心時間序列指標
    if 'train_loss' in hyperparams:
        metrics['train_loss'] = float(hyperparams['train_loss'])
    if 'val_loss' in hyperparams:
        metrics['val_loss'] = float(hyperparams['val_loss'])
    if 'overfitting_ratio' in hyperparams:
        metrics['overfitting_ratio'] = float(hyperparams['overfitting_ratio'])
    
    # 預測誤差指標
    if 'mase' in hyperparams:
        metrics['mase'] = float(hyperparams['mase'])
    if 'mae' in hyperparams:
        metrics['mae'] = float(hyperparams.get('original_mae', hyperparams['mae']))
    if 'rmse' in hyperparams:
        metrics['rmse'] = float(hyperparams.get('original_rmse', hyperparams['rmse']))
    if 'smape' in hyperparams:
        metrics['smape'] = float(hyperparams.get('original_smape', hyperparams['smape']))
    if 'mse' in hyperparams:
        metrics['mse'] = float(hyperparams['mse'])
    if 'mape' in hyperparams:
        metrics['mape'] = float(hyperparams['mape'])
    if 'msmape' in hyperparams:
        metrics['msmape'] = float(hyperparams['msmape'])
    
    return metrics


# ============================================================================
# 步驟 2: 定義自訂的reward函數
# ============================================================================

def timeseries_reward_function(metrics: dict) -> float:
    """
    基於訓練損失計算reward（避免data leakage）
    
    使用訓練損失而非測試集指標，確保符合benchmark要求
    
    Args:
        metrics: 包含所有評估指標的字典
        
    Returns:
        reward值（越大越好）
    """
    train_loss = metrics.get('train_loss', 1.0)
    
    # 防止無效值
    if np.isnan(train_loss) or np.isinf(train_loss) or train_loss <= 0:
        train_loss = 1.0
    
    # 將訓練損失轉換為reward（越小的loss得到越高的reward）
    # 使用負對數轉換
    pseudo_mase = -np.log(max(train_loss, 1e-6))
    pseudo_mase = max(0.1, min(10.0, pseudo_mase))
    
    # 轉換為0-1範圍的reward
    if pseudo_mase <= 0.5:
        reward = 0.8 - 0.15 * (np.log(pseudo_mase + 0.1) + 2.3)
    elif pseudo_mase <= 1.0:
        normalized = (pseudo_mase - 0.5) / 0.5
        reward = 0.72 - 0.27 * normalized
    elif pseudo_mase <= 2.0:
        normalized = (pseudo_mase - 1.0) / 1.0
        reward = 0.45 * np.exp(-normalized * 0.8)
    else:
        reward = 0.2 * np.exp(-(pseudo_mase - 2.0) * 0.3)
    
    return max(0.05, min(0.9, reward))


# ============================================================================
# 步驟 3: 使用通用接口創建環境
# ============================================================================

class TimeSeriesEnvironmentWithCustomMetrics(BaseEnvironment):
    """
    使用通用接口的時間序列環境
    
    不需要修改MAT-HPO-Library的代碼，直接通過參數配置
    """
    
    def __init__(self, model_name: str, dataset_name: str, **kwargs):
        # 定義要追蹤的custom metrics
        custom_metrics = ['train_loss', 'val_loss', 'mase', 'smape', 'mae', 'rmse', 'overfitting_ratio']
        
        # 定義metric名稱映射（用於顯示）
        metric_names_mapping = {
            'f1': 'SMAPE',
            'auc': 'MAE', 
            'gmean': 'RMSE'
        }
        
        # 使用自訂reward函數
        super().__init__(
            name=f"TimeSeries-{model_name}-{dataset_name}",
            custom_metrics=custom_metrics,
            metric_names_mapping=metric_names_mapping,
            reward_function=timeseries_reward_function,
            **kwargs
        )
        
        self.model_name = model_name
        self.dataset_name = dataset_name
    
    def load_data(self):
        """載入數據"""
        # 實作數據載入邏輯
        pass
    
    def create_model(self, hyperparams):
        """創建模型"""
        # 實作模型創建邏輯
        pass
    
    def train_evaluate(self, model, hyperparams):
        """訓練和評估"""
        # 訓練模型...
        
        # 返回所有原始指標
        return {
            'train_loss': 331.72,
            'val_loss': 346.51,
            'overfitting_ratio': 1.045,
            'mase': 2.304,
            'smape': 0.0618,
            'mae': 632.53,
            'rmse': 809.25,
            'mse': 654889.81,
            # 轉換後的值（給MAT-HPO優化用）
            'f1': 0.7691,   # SMAPE轉換後
            'auc': 0.1675,  # MAE轉換後
            'gmean': 0.1000, # RMSE轉換後
            # 保存原始值
            'original_smape': 0.0618,
            'original_mae': 632.53,
            'original_rmse': 809.25
        }
    
    def compute_reward(self, metrics):
        """計算reward"""
        # 如果提供了custom_reward_function，使用它
        if hasattr(self, 'custom_reward_function') and self.custom_reward_function:
            return self.custom_reward_function(metrics)
        else:
            # 默認使用train_loss
            return timeseries_reward_function(metrics)


# ============================================================================
# 步驟 4: 使用通用接口創建optimizer和logger
# ============================================================================

def main():
    """主函數：展示如何使用通用接口"""
    
    # 創建環境（使用custom metrics和reward）
    env = TimeSeriesEnvironmentWithCustomMetrics(
        model_name="dlinear",
        dataset_name="us_births"
    )
    
    # 創建hyperparameter space
    space = HyperparameterSpace()
    space.add_continuous('learning_rate', 1e-5, 1e-2, agent=0)
    space.add_discrete('batch_size', [8, 16, 32, 64], agent=0)
    
    # 創建配置
    config = DefaultConfigs.standard()
    config.max_steps = 5
    
    # 創建logger（使用custom metrics extractor）
    logger = HPOLogger(
        output_dir='./timeseries_custom_test',
        metric_names={'f1': 'SMAPE', 'auc': 'MAE', 'gmean': 'RMSE'},
        custom_metrics=['train_loss', 'val_loss', 'mase', 'smape', 'mae', 'rmse'],
        metrics_extractor=timeseries_metrics_extractor
    )
    
    # 創建optimizer
    optimizer = MAT_HPO_Optimizer(env, space, config)
    optimizer.logger = logger  # 使用自訂logger
    
    # 運行優化
    results = optimizer.optimize()
    
    print("\n✅ 優化完成！")
    print(f"最佳reward: {results['best_performance']['reward']:.4f}")
    print(f"最佳超參數: {results['best_hyperparameters']}")


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 MAT-HPO 通用接口示例：時間序列預測")
    print("=" * 60)
    print("\n📝 展示功能：")
    print("  1. 自訂metrics列表")
    print("  2. 自訂metrics提取器")
    print("  3. 自訂reward函數")
    print("  4. 自訂metric名稱映射")
    print("\n" + "=" * 60 + "\n")
    
    main()


