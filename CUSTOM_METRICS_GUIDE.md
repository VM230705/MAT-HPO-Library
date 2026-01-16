# MAT-HPO Library - Custom Metrics Guide

## 設計理念

MAT-HPO-Library提供**通用接口**，讓使用者可以透過參數傳入自訂的metrics和reward設計，**而不需要修改庫本身的代碼**。

## 通用接口概覽

### 1. **BaseEnvironment 自訂參數**

```python
class MyEnvironment(BaseEnvironment):
    def __init__(self, ...):
        super().__init__(
            name="MyEnvironment",
            # 🎯 自訂metrics列表
            custom_metrics=['train_loss', 'val_loss', 'mase', 'smape', 'mae', 'rmse'],
            
            # 📊 Metric名稱映射（用於顯示）
            metric_names_mapping={
                'f1': 'SMAPE',    # F1顯示為SMAPE
                'auc': 'MAE',     # AUC顯示為MAE
                'gmean': 'RMSE'   # G-mean顯示為RMSE
            },
            
            # 🎁 自訂reward函數
            reward_function=my_custom_reward_function
        )
```

### 2. **HPOLogger 自訂參數**

```python
from MAT_HPO_LIB.utils.logger import HPOLogger

# 定義metrics提取器
def extract_timeseries_metrics(hyperparams: dict) -> dict:
    """從hyperparams中提取要記錄的指標"""
    return {
        'train_loss': float(hyperparams.get('train_loss', 0.0)),
        'val_loss': float(hyperparams.get('val_loss', 0.0)),
        'mase': float(hyperparams.get('mase', 1.0)),
        'smape': float(hyperparams.get('original_smape', 0.0)),
        'mae': float(hyperparams.get('original_mae', 0.0)),
        'rmse': float(hyperparams.get('original_rmse', 0.0)),
    }

# 創建自訂logger
logger = HPOLogger(
    output_dir='./results',
    metric_names={'f1': 'SMAPE', 'auc': 'MAE', 'gmean': 'RMSE'},
    custom_metrics=['train_loss', 'val_loss', 'mase', 'smape', 'mae', 'rmse'],
    metrics_extractor=extract_timeseries_metrics
)
```

## 🚀 完整使用示例

### **時間序列預測場景**

```python
#!/usr/bin/env python3
import numpy as np
from MAT_HPO_LIB import MAT_HPO_Optimizer, BaseEnvironment, HyperparameterSpace
from MAT_HPO_LIB.utils.logger import HPOLogger

# ============================================================================
# 1. 定義自訂reward函數
# ============================================================================

def timeseries_reward(metrics: dict) -> float:
    """基於訓練損失計算reward（避免data leakage）"""
    train_loss = metrics.get('train_loss', 1.0)
    if train_loss <= 0:
        train_loss = 1.0
    
    # 轉換為reward（越小的loss越高的reward）
    reward = -np.log(max(train_loss, 1e-6))
    reward = max(0.1, min(10.0, reward))
    
    # 映射到0-1範圍
    reward = max(0.05, min(0.9, reward / 10.0))
    return reward

# ============================================================================
# 2. 定義metrics提取器
# ============================================================================

def extract_metrics(hyperparams: dict) -> dict:
    """從hyperparams提取所有要記錄的原始指標"""
    metrics = {
        'train_loss': float(hyperparams.get('train_loss', 0.0)),
        'mase': float(hyperparams.get('mase', 0.0)),
        'smape': float(hyperparams.get('original_smape', 0.0)),
        'mae': float(hyperparams.get('original_mae', 0.0)),
        'rmse': float(hyperparams.get('original_rmse', 0.0)),
    }
    
    # 可選的validation指標
    if 'val_loss' in hyperparams:
        metrics['val_loss'] = float(hyperparams['val_loss'])
    if 'overfitting_ratio' in hyperparams:
        metrics['overfitting_ratio'] = float(hyperparams['overfitting_ratio'])
    
    return metrics

# ============================================================================
# 3. 創建環境（使用自訂配置）
# ============================================================================

class TimeSeriesEnvironment(BaseEnvironment):
    def __init__(self, model_name, dataset_name):
        super().__init__(
            name=f"TS-{model_name}-{dataset_name}",
            custom_metrics=['train_loss', 'val_loss', 'mase', 'smape', 'mae', 'rmse'],
            metric_names_mapping={'f1': 'SMAPE', 'auc': 'MAE', 'gmean': 'RMSE'},
            reward_function=timeseries_reward
        )
        self.model_name = model_name
        self.dataset_name = dataset_name
    
    def load_data(self):
        # 載入時間序列數據
        pass
    
    def create_model(self, hyperparams):
        # 創建模型
        pass
    
    def train_evaluate(self, model, hyperparams):
        # 訓練模型...
        train_loss = 331.72  # 示例值
        val_loss = 346.51
        
        # 評估模型...
        mase = 2.304
        mae = 632.53
        rmse = 809.25
        smape = 0.0618
        
        # 返回所有指標
        return {
            # 原始訓練指標
            'train_loss': train_loss,
            'val_loss': val_loss,
            'overfitting_ratio': val_loss / train_loss,
            
            # 原始測試集指標
            'mase': mase,
            'smape': smape,
            'mae': mae,
            'rmse': rmse,
            'mse': rmse ** 2,
            
            # 轉換後的值（給MAT-HPO優化用）
            # 將"越小越好"的指標轉換為"越大越好"
            'f1': 0.8 - min(0.8, smape / 2.0),
            'auc': 0.8 - min(0.8, mae / 1000.0),
            'gmean': 0.8 - min(0.8, rmse / 1000.0),
            
            # 保存原始值（使用original_前綴）
            'original_smape': smape,
            'original_mae': mae,
            'original_rmse': rmse
        }
    
    def compute_reward(self, metrics):
        # 使用自訂reward函數
        if self.custom_reward_function:
            return self.custom_reward_function(metrics)
        else:
            return timeseries_reward(metrics)

# ============================================================================
# 4. 設置optimizer和logger
# ============================================================================

def run_optimization():
    # 創建環境
    env = TimeSeriesEnvironment("dlinear", "us_births")
    
    # 創建hyperparameter space
    space = HyperparameterSpace()
    space.add_continuous('learning_rate', 1e-5, 1e-2, agent=0)
    space.add_discrete('batch_size', [8, 16, 32, 64], agent=0)
    space.add_discrete('epochs', [20, 50, 100], agent=0)
    
    # 創建配置
    config = DefaultConfigs.standard()
    config.max_steps = 10
    
    # 創建自訂logger
    logger = HPOLogger(
        output_dir='./timeseries_results',
        metric_names={'f1': 'SMAPE', 'auc': 'MAE', 'gmean': 'RMSE'},
        custom_metrics=['train_loss', 'val_loss', 'mase', 'smape', 'mae', 'rmse'],
        metrics_extractor=extract_metrics  # 使用自訂提取器
    )
    
    # 創建optimizer
    optimizer = MAT_HPO_Optimizer(env, space, config)
    optimizer.logger = logger  # 設置自訂logger
    
    # 運行優化
    results = optimizer.optimize()
    
    return results

if __name__ == "__main__":
    results = run_optimization()
    print(f"\n✅ 最佳reward: {results['best_performance']['reward']:.4f}")
```

## 輸出格式

### **step_log.jsonl 格式**

使用自訂metrics extractor後，每個步驟會記錄：

```json
{
  "step": 0,
  "timestamp": "2025-10-01T08:00:00",
  "metrics": {
    "train_loss": 331.72,
    "val_loss": 346.51,
    "overfitting_ratio": 1.045,
    "mase": 2.304,
    "smape": 0.0618,
    "mae": 632.53,
    "rmse": 809.25,
    "mse": 654889.81,
    "f1_transformed": 0.7691,
    "auc_transformed": 0.1675,
    "gmean_transformed": 0.1000
  },
  "timing": {...},
  "hyperparameters": {...}
}
```

## ✅ 優點

1. **✅ 不需要修改MAT-HPO-Library代碼** - 通過參數配置
2. **✅ 完全自訂metrics** - 追蹤任何想要的指標
3. **✅ 自訂reward邏輯** - 靈活定義優化目標
4. **✅ 保持原始值** - 記錄未經轉換的真實數據
5. **✅ 通用可重用** - 適用於任何領域（時間序列、分類、回歸等）

## 使用建議

1. **定義metrics_extractor**: 明確指定要記錄哪些原始指標
2. **定義reward_function**: 明確如何計算reward
3. **使用custom_metrics**: 列出所有要追蹤的指標名稱
4. **使用metric_names_mapping**: 將內部名稱映射到易讀的顯示名稱

這樣每次使用MAT-HPO-Library時，只需要在**自己的項目代碼中**定義這些函數和配置，而不需要修改庫本身！


