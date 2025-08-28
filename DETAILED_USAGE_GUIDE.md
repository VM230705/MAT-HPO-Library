# MAT-HPO Library - 詳細使用說明

## 📖 目錄
1. [概述](#概述)
2. [安裝與設定](#安裝與設定)
3. [核心概念](#核心概念)
4. [快速開始](#快速開始)
5. [詳細教學](#詳細教學)
6. [進階功能](#進階功能)
7. [完整範例](#完整範例)
8. [常見問題](#常見問題)
9. [最佳實踐](#最佳實踐)

## 概述

MAT-HPO (Multi-Agent Transformer Hyperparameter Optimization) 是一個強大而靈活的超參數優化函式庫，專為複雜的機器學習問題設計。它使用多代理強化學習方法，讓不同的代理專門負責不同類型的超參數優化。

### 🌟 主要特點

- **多代理架構**: 三個專門的代理分別優化不同類型的超參數
- **Transformer網路**: 使用先進的Transformer架構進行智能超參數選擇
- **彈性整合**: 易於整合到任何機器學習管道中
- **強大的驗證**: 完整的錯誤檢查和參數驗證機制
- **豐富的功能**: 支援檢查點、回調函數、自訂約束等
- **詳細日誌**: 完整的優化歷史記錄和效能追蹤

### 🎯 適用場景

- 深度學習模型超參數調整
- 傳統機器學習算法優化
- AutoML系統開發
- 研究實驗自動化
- 多目標優化問題

## 安裝與設定

### 基本需求

```bash
# Python 3.8+
pip install torch numpy scikit-learn tqdm
```

### 安裝函式庫

```bash
# 方法1: 直接從原始碼使用
cd path/to/MAT_HPO_LIB
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 方法2: 安裝為Python包 (推薦)
pip install -e .
```

### 驗證安裝

```python
from MAT_HPO_LIB import MAT_HPO_Optimizer, BaseEnvironment, HyperparameterSpace
print("✅ MAT-HPO Library successfully imported!")
```

## 核心概念

### 1. 多代理分工

MAT-HPO使用三個專門的代理來優化不同類型的超參數：

```python
# 典型的代理分工
Agent 0: 問題特定參數 (類別權重、正規化、領域相關參數)
Agent 1: 模型架構參數 (隱藏層大小、層數、網路結構)  
Agent 2: 訓練參數 (批次大小、學習率、優化器設定)
```

### 2. 核心組件

#### BaseEnvironment (基礎環境)
```python
class MyEnvironment(BaseEnvironment):
    def load_data(self):
        # 加載和預處理數據
        return data
    
    def create_model(self, hyperparams):
        # 根據超參數創建模型
        return model
    
    def train_evaluate(self, model, hyperparams):
        # 訓練模型並評估效能
        return {'accuracy': 0.95, 'f1': 0.93}
    
    def compute_reward(self, metrics):
        # 計算獎勵信號
        return metrics['f1'] * 0.6 + metrics['accuracy'] * 0.4
```

#### HyperparameterSpace (超參數空間)
```python
space = HyperparameterSpace(
    agent0_params=['class_weight_0', 'class_weight_1'],
    agent1_params=['hidden_size', 'num_layers'],
    agent2_params=['batch_size', 'learning_rate'],
    bounds={
        'class_weight_0': (0.1, 5.0),      # 數值範圍
        'hidden_size': (64, 512),          # 整數範圍
        'learning_rate': (1e-5, 1e-2),     # 對數分布
        'optimizer': ['adam', 'sgd']       # 類別選擇
    },
    param_types={
        'class_weight_0': float,
        'hidden_size': int,
        'learning_rate': 'log_uniform',
        'optimizer': str
    }
)
```

#### MAT_HPO_Optimizer (優化器)
```python
optimizer = MAT_HPO_Optimizer(
    environment=environment,
    hyperparameter_space=hyperparameter_space,
    config=OptimizationConfig(
        max_steps=100,
        use_cuda=True,
        verbose=True
    )
)
```

## 快速開始

### 30秒快速範例

```python
from MAT_HPO_LIB import *
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

class SimpleNNEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__(name="SimpleNN")
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
    
    def load_data(self):
        # 生成示例數據
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return {"loaded": True}
    
    def create_model(self, hyperparams):
        model = nn.Sequential(
            nn.Linear(20, hyperparams['hidden_size']),
            nn.ReLU(),
            nn.Linear(hyperparams['hidden_size'], 2)
        )
        return model
    
    def train_evaluate(self, model, hyperparams):
        # 簡化的訓練過程
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # 快速訓練
        X_train = torch.FloatTensor(self.X_train)
        y_train = torch.LongTensor(self.y_train)
        X_val = torch.FloatTensor(self.X_val)
        y_val = torch.LongTensor(self.y_val)
        
        for epoch in range(10):  # 快速示例，只訓練10個epoch
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        # 評估
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            _, predicted = torch.max(val_outputs.data, 1)
            accuracy = accuracy_score(y_val, predicted)
            f1 = f1_score(y_val, predicted)
        
        return {'accuracy': accuracy, 'f1': f1}
    
    def compute_reward(self, metrics):
        return metrics['f1'] * 0.7 + metrics['accuracy'] * 0.3

# 定義超參數空間
space = HyperparameterSpace(
    agent0_params=[],  # 這個例子中沒有問題特定參數
    agent1_params=['hidden_size'],
    agent2_params=['learning_rate'],
    bounds={
        'hidden_size': (32, 256),
        'learning_rate': (1e-4, 1e-2)
    },
    param_types={
        'hidden_size': int,
        'learning_rate': 'log_uniform'
    }
)

# 創建和運行優化器
environment = SimpleNNEnvironment()
optimizer = MAT_HPO_Optimizer(environment, space, DefaultConfigs.quick_test())
results = optimizer.optimize()

print(f"🎉 最佳超參數: {results['best_hyperparameters']}")
print(f"🏆 最佳效能: F1={results['best_performance']['f1']:.4f}")
```

## 詳細教學

### 1. 環境設計詳解

#### 數據加載策略
```python
class AdvancedEnvironment(BaseEnvironment):
    def load_data(self):
        # 方式1: 從文件加載
        train_data = pd.read_csv('train.csv')
        val_data = pd.read_csv('val.csv')
        
        # 方式2: 使用數據增強
        train_data = self.apply_augmentation(train_data)
        
        # 方式3: 支援多種數據格式
        if self.data_type == 'images':
            return self.load_image_data()
        elif self.data_type == 'text':
            return self.load_text_data()
        else:
            return self.load_tabular_data()
    
    def load_image_data(self):
        # 圖像數據加載邏輯
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = ImageFolder('data/train', transform=transform)
        val_dataset = ImageFolder('data/val', transform=transform)
        
        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset
        }
```

#### 模型創建最佳實踐
```python
def create_model(self, hyperparams):
    # 根據問題類型選擇基礎架構
    if self.problem_type == 'classification':
        return self._create_classifier(hyperparams)
    elif self.problem_type == 'regression':
        return self._create_regressor(hyperparams)
    elif self.problem_type == 'time_series':
        return self._create_time_series_model(hyperparams)

def _create_classifier(self, hyperparams):
    # 支援動態架構
    layers = []
    input_size = self.feature_dim
    
    # 動態添加隱藏層
    for i in range(hyperparams.get('num_layers', 2)):
        hidden_size = hyperparams.get(f'hidden_size_{i}', 128)
        layers.extend([
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(hyperparams.get('dropout', 0.5))
        ])
        input_size = hidden_size
    
    # 輸出層
    layers.append(nn.Linear(input_size, self.num_classes))
    
    return nn.Sequential(*layers)
```

#### 評估和獎勵設計
```python
def train_evaluate(self, model, hyperparams):
    # 完整的訓練循環
    train_loader = DataLoader(self.train_dataset, 
                             batch_size=hyperparams['batch_size'],
                             shuffle=True)
    val_loader = DataLoader(self.val_dataset, 
                           batch_size=hyperparams['batch_size'])
    
    # 優化器選擇
    optimizer_type = hyperparams.get('optimizer', 'adam')
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=hyperparams['learning_rate'])
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                  lr=hyperparams['learning_rate'],
                                  momentum=hyperparams.get('momentum', 0.9))
    
    # 學習率調度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=hyperparams.get('lr_step_size', 10),
        gamma=hyperparams.get('lr_gamma', 0.1)
    )
    
    # 訓練循環
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(hyperparams.get('epochs', 50)):
        # 訓練階段
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = F.cross_entropy(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            if hyperparams.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                             hyperparams['gradient_clip'])
            
            optimizer.step()
            train_loss += loss.item()
        
        # 驗證階段
        model.eval()
        val_loss, predictions, targets = self._validate(model, val_loader)
        
        # 早停檢查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            
        if patience_counter >= hyperparams.get('early_stopping_patience', 10):
            model.load_state_dict(best_model)
            break
        
        scheduler.step()
    
    # 計算最終指標
    metrics = self._compute_metrics(predictions, targets)
    metrics['training_epochs'] = epoch + 1
    metrics['final_train_loss'] = train_loss / len(train_loader)
    metrics['final_val_loss'] = best_val_loss
    
    return metrics

def compute_reward(self, metrics):
    # 多目標獎勵函數
    primary_score = metrics.get('f1', metrics.get('accuracy', 0))
    
    # 效率獎勵 (訓練時間越短越好)
    efficiency_bonus = max(0, 1.0 - metrics.get('training_epochs', 50) / 50)
    
    # 穩定性獎勵 (驗證損失越低越好)
    stability_bonus = max(0, 1.0 - metrics.get('final_val_loss', 1.0))
    
    # 複雜度懲罰 (避免過度複雜的模型)
    complexity_penalty = 0.1 * metrics.get('model_params', 0) / 1000000  # 每百萬參數0.1的懲罰
    
    total_reward = (primary_score * 0.7 + 
                   efficiency_bonus * 0.15 + 
                   stability_bonus * 0.1 + 
                   complexity_penalty * 0.05)
    
    return total_reward
```

### 2. 超參數空間設計

#### 支援的參數類型
```python
# 數值參數
'learning_rate': (1e-5, 1e-2)        # 線性分布
'learning_rate': 'log_uniform'        # 對數分布 (推薦用於學習率)

# 整數參數  
'hidden_size': (64, 512)              # 整數範圍
'num_layers': (1, 10)                 # 層數

# 類別參數
'optimizer': ['adam', 'sgd', 'rmsprop', 'adamw']
'activation': ['relu', 'gelu', 'swish', 'leaky_relu']

# 布林參數
'use_batch_norm': bool                # 自動設為 (False, True)
'use_dropout': (False, True)          # 明確指定
```

#### 複雜超參數空間範例
```python
# 完整的深度學習超參數空間
space = HyperparameterSpace(
    agent0_params=[
        'class_weight_0', 'class_weight_1', 'class_weight_2',  # 類別平衡
        'data_augmentation_strength',                          # 數據增強
        'regularization_strength'                              # 正規化
    ],
    agent1_params=[
        'backbone_type',           # 主幹網路類型
        'hidden_size_1',          # 第一隱藏層
        'hidden_size_2',          # 第二隱藏層  
        'num_layers',             # 總層數
        'activation_type',        # 激活函數
        'use_batch_norm',         # 是否使用BN
        'dropout_rate'            # Dropout比率
    ],
    agent2_params=[
        'batch_size',             # 批次大小
        'learning_rate',          # 學習率
        'optimizer_type',         # 優化器
        'lr_schedule',            # 學習率調度
        'weight_decay',           # 權重衰減
        'gradient_clip_norm',     # 梯度裁剪
        'early_stopping_patience' # 早停耐心
    ],
    bounds={
        # Agent 0: 問題特定參數
        'class_weight_0': (0.1, 10.0),
        'class_weight_1': (0.1, 10.0), 
        'class_weight_2': (0.1, 10.0),
        'data_augmentation_strength': (0.0, 1.0),
        'regularization_strength': (1e-6, 1e-2),
        
        # Agent 1: 架構參數
        'backbone_type': ['resnet', 'densenet', 'efficientnet'],
        'hidden_size_1': (64, 1024),
        'hidden_size_2': (32, 512),
        'num_layers': (2, 8),
        'activation_type': ['relu', 'gelu', 'swish'],
        'use_batch_norm': (False, True),
        'dropout_rate': (0.0, 0.8),
        
        # Agent 2: 訓練參數
        'batch_size': (8, 128),
        'learning_rate': (1e-5, 1e-1),
        'optimizer_type': ['adam', 'adamw', 'sgd', 'rmsprop'],
        'lr_schedule': ['constant', 'step', 'cosine', 'exponential'],
        'weight_decay': (1e-6, 1e-2),
        'gradient_clip_norm': (0.1, 10.0),
        'early_stopping_patience': (5, 50)
    },
    param_types={
        # Agent 0
        'class_weight_0': float,
        'class_weight_1': float,
        'class_weight_2': float,
        'data_augmentation_strength': float,
        'regularization_strength': 'log_uniform',
        
        # Agent 1  
        'backbone_type': str,
        'hidden_size_1': int,
        'hidden_size_2': int,
        'num_layers': int,
        'activation_type': str,
        'use_batch_norm': bool,
        'dropout_rate': float,
        
        # Agent 2
        'batch_size': int,
        'learning_rate': 'log_uniform',
        'optimizer_type': str,
        'lr_schedule': str,
        'weight_decay': 'log_uniform',
        'gradient_clip_norm': float,
        'early_stopping_patience': int
    },
    default_values={
        'learning_rate': 1e-3,
        'batch_size': 32,
        'optimizer_type': 'adam',
        'activation_type': 'relu'
    },
    parameter_descriptions={
        'learning_rate': '學習率，控制模型更新步長',
        'batch_size': '批次大小，影響梯度估計質量',
        'hidden_size_1': '第一隱藏層神經元數量'
    }
)
```

### 3. 優化配置和回調

#### 優化配置詳解
```python
from MAT_HPO_LIB.utils.config import OptimizationConfig, DefaultConfigs

# 自定義配置
config = OptimizationConfig(
    # 核心設定
    max_steps=200,                    # 最大優化步數
    replay_buffer_size=2000,          # 經驗回放緩衝區大小
    batch_size=64,                    # SQDDPG批次大小
    
    # 學習率設定
    policy_learning_rate=1e-4,        # 策略網路學習率
    value_learning_rate=1e-3,         # 價值網路學習率
    
    # 更新頻率
    behaviour_update_freq=5,          # 行為策略更新頻率
    critic_update_times=2,            # 評價者更新次數
    
    # 設備設定
    gpu_device=0,                     # GPU設備編號
    use_cuda=True,                    # 是否使用CUDA
    
    # 日誌和保存
    save_interval=20,                 # 模型保存間隔
    log_interval=1,                   # 日誌輸出間隔
    verbose=True,                     # 詳細輸出
    
    # 早停設定
    early_stop_patience=30,           # 早停耐心值
    early_stop_threshold=1e-4,        # 早停閾值
    
    # 進階設定
    gradient_clip=1.0,                # 梯度裁剪
    target_update_tau=0.005,          # 目標網路更新率
    noise_std=0.1,                    # 探索噪音標準差
    
    # 隨機種子
    seed=42,                          # 隨機種子
    deterministic=True                # 確定性執行
)

# 或使用預設配置
config = DefaultConfigs.standard()   # 標準配置
config = DefaultConfigs.quick_test()  # 快速測試
config = DefaultConfigs.extensive()  # 廣泛搜索
config = DefaultConfigs.cpu_only()   # CPU專用
```

#### 回調函數系統
```python
def step_callback(env, hyperparams, metrics, reward):
    """每步執行的回調函數"""
    print(f"Step {env.current_step}: Reward = {reward:.4f}")
    
    # 記錄到外部系統
    wandb.log({
        'reward': reward,
        'f1_score': metrics.get('f1', 0),
        'step': env.current_step
    })
    
    # 自定義早停條件
    if reward > 0.95:
        print("🎯 達到目標效能，建議早停")

def epoch_callback(env, epoch, model, metrics):
    """每個epoch執行的回調函數"""
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {metrics.get('loss', 0):.4f}")

# 添加回調
environment.add_step_callback(step_callback)
environment.add_epoch_callback(epoch_callback)
```

### 4. 檢查點和恢復

```python
# 啟用檢查點
environment = MyEnvironment(
    checkpoint_dir="./checkpoints/experiment_1",
    verbose=True,
    save_history=True
)

# 運行優化
optimizer = MAT_HPO_Optimizer(environment, space, config, 
                             output_dir="./results/experiment_1")
results = optimizer.optimize()

# 恢復優化 (如果中斷)
if os.path.exists("./checkpoints/experiment_1"):
    environment.load_checkpoint("./checkpoints/experiment_1")
    optimizer.load_checkpoint("./checkpoints/experiment_1")
    results = optimizer.optimize()  # 繼續優化
```

## 進階功能

### 1. 自訂約束函數

```python
def model_size_constraint(hyperparams):
    """限制模型大小的約束函數"""
    total_params = hyperparams['hidden_size_1'] * hyperparams['hidden_size_2']
    if total_params > 100000:  # 限制在10萬參數以內
        # 等比例縮小
        scale = (100000 / total_params) ** 0.5
        hyperparams['hidden_size_1'] = int(hyperparams['hidden_size_1'] * scale)
        hyperparams['hidden_size_2'] = int(hyperparams['hidden_size_2'] * scale)
    return hyperparams

def batch_lr_constraint(hyperparams):
    """批次大小和學習率的約束"""
    # 大批次使用大學習率
    if hyperparams['batch_size'] > 64:
        hyperparams['learning_rate'] = max(hyperparams['learning_rate'], 1e-3)
    return hyperparams

# 添加約束
space.add_constraint(model_size_constraint)
space.add_constraint(batch_lr_constraint)
```

### 2. 多目標優化

```python
class MultiObjectiveEnvironment(BaseEnvironment):
    def compute_reward(self, metrics):
        # Pareto前沿優化
        accuracy = metrics.get('accuracy', 0)
        inference_speed = metrics.get('inference_speed', 0)
        model_size = metrics.get('model_size', 1000000)
        
        # 歸一化指標
        norm_accuracy = accuracy  # 已經是0-1範圍
        norm_speed = min(inference_speed / 1000, 1.0)  # 歸一化到0-1
        norm_size = max(0, 1.0 - model_size / 10000000)  # 越小越好
        
        # 加權組合 (可根據需求調整權重)
        weights = [0.5, 0.3, 0.2]  # [準確率, 速度, 大小]
        total_reward = (weights[0] * norm_accuracy + 
                       weights[1] * norm_speed + 
                       weights[2] * norm_size)
        
        return total_reward
```

### 3. 分佈式優化

```python
import torch.distributed as dist
import torch.multiprocessing as mp

class DistributedEnvironment(BaseEnvironment):
    def __init__(self, rank, world_size):
        super().__init__(name=f"DistributedEnv-{rank}")
        self.rank = rank
        self.world_size = world_size
        
    def train_evaluate(self, model, hyperparams):
        # 分佈式訓練
        model = torch.nn.parallel.DistributedDataParallel(model)
        
        # 使用分佈式數據加載器
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset, num_replicas=self.world_size, rank=self.rank
        )
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=hyperparams['batch_size'],
            sampler=sampler
        )
        
        # 分佈式訓練邏輯
        return self._distributed_training(model, train_loader, hyperparams)

def run_distributed_optimization(rank, world_size):
    # 初始化分佈式環境
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    environment = DistributedEnvironment(rank, world_size)
    optimizer = MAT_HPO_Optimizer(environment, space, config)
    
    results = optimizer.optimize()
    
    # 清理
    dist.destroy_process_group()
    return results

# 啟動分佈式優化
if __name__ == '__main__':
    world_size = 4
    mp.spawn(run_distributed_optimization, args=(world_size,), nprocs=world_size)
```

### 4. 自適應搜索空間

```python
class AdaptiveHyperparameterSpace(HyperparameterSpace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_history = []
        
    def update_bounds_based_on_performance(self, hyperparams, performance):
        """根據效能動態調整搜索空間"""
        self.performance_history.append((hyperparams.copy(), performance))
        
        if len(self.performance_history) >= 10:  # 至少10個樣本
            # 找出效能前25%的配置
            sorted_history = sorted(self.performance_history, 
                                  key=lambda x: x[1], reverse=True)
            top_configs = sorted_history[:len(sorted_history)//4]
            
            # 縮小搜索範圍到高效能區域
            for param in self.agent1_params + self.agent2_params:
                if param in self.numerical_params:
                    values = [config[0][param] for config, _ in top_configs 
                             if param in config[0]]
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        
                        # 更新搜索範圍 (mean ± 2*std)
                        new_min = max(self.bounds[param][0], mean_val - 2*std_val)
                        new_max = min(self.bounds[param][1], mean_val + 2*std_val)
                        
                        self.bounds[param] = (new_min, new_max)
                        print(f"📊 調整 {param} 搜索範圍到 [{new_min:.4f}, {new_max:.4f}]")
```

## 完整範例

### 範例1: 圖像分類優化

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

class CIFAR10Environment(BaseEnvironment):
    def __init__(self):
        super().__init__(name="CIFAR10-Classification")
        self.num_classes = 10
        self.input_channels = 3
        
    def load_data(self):
        # 數據轉換
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # 加載數據集
        train_dataset = CIFAR10(root='./data', train=True, 
                               transform=train_transform, download=True)
        val_dataset = CIFAR10(root='./data', train=False, 
                             transform=val_transform)
        
        # 使用子集以加速實驗
        train_size = len(train_dataset) // 10  # 使用1/10數據
        val_size = len(val_dataset) // 10
        
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(val_dataset, range(val_size))
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        return {"train_size": train_size, "val_size": val_size}
    
    def create_model(self, hyperparams):
        """創建CNN模型"""
        class SimpleCNN(nn.Module):
            def __init__(self, hyperparams):
                super().__init__()
                
                # 卷積層
                self.conv1 = nn.Conv2d(3, hyperparams['conv1_filters'], 3, padding=1)
                self.conv2 = nn.Conv2d(hyperparams['conv1_filters'], 
                                     hyperparams['conv2_filters'], 3, padding=1)
                
                # 批次正規化 (可選)
                self.bn1 = nn.BatchNorm2d(hyperparams['conv1_filters'])
                self.bn2 = nn.BatchNorm2d(hyperparams['conv2_filters'])
                self.use_batch_norm = hyperparams.get('use_batch_norm', True)
                
                # 全連接層
                fc_input_size = hyperparams['conv2_filters'] * 8 * 8  # 32->16->8
                self.fc1 = nn.Linear(fc_input_size, hyperparams['hidden_size'])
                self.fc2 = nn.Linear(hyperparams['hidden_size'], 10)
                
                # Dropout
                self.dropout = nn.Dropout(hyperparams.get('dropout_rate', 0.5))
                
            def forward(self, x):
                # 卷積 + 池化
                x = self.conv1(x)
                if self.use_batch_norm:
                    x = self.bn1(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2)
                
                x = self.conv2(x)
                if self.use_batch_norm:
                    x = self.bn2(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2)
                
                # 展平
                x = x.view(x.size(0), -1)
                
                # 全連接層
                x = self.dropout(torch.relu(self.fc1(x)))
                x = self.fc2(x)
                
                return x
        
        return SimpleCNN(hyperparams)
    
    def train_evaluate(self, model, hyperparams):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 數據加載器
        train_loader = DataLoader(self.train_dataset, 
                                 batch_size=hyperparams['batch_size'],
                                 shuffle=True, num_workers=2)
        val_loader = DataLoader(self.val_dataset,
                               batch_size=hyperparams['batch_size'],
                               shuffle=False, num_workers=2)
        
        # 優化器和損失函數
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=hyperparams['learning_rate'],
                                   weight_decay=hyperparams.get('weight_decay', 1e-4))
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # 訓練
        best_val_acc = 0
        epochs = hyperparams.get('epochs', 20)
        
        for epoch in range(epochs):
            # 訓練階段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            # 驗證階段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # 計算準確率
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            best_val_acc = max(best_val_acc, val_acc)
            
            # 早停檢查
            if epoch > 5 and val_acc < best_val_acc * 0.95:  # 簡單早停策略
                break
                
            scheduler.step()
        
        return {
            'accuracy': best_val_acc,
            'final_train_acc': train_acc,
            'final_val_acc': val_acc,
            'epochs_trained': epoch + 1
        }
    
    def compute_reward(self, metrics):
        # 主要關注驗證準確率，但也考慮效率
        val_acc = metrics['accuracy']
        efficiency_bonus = max(0, 1.0 - metrics['epochs_trained'] / 20)  # 訓練越快獎勵越多
        
        return val_acc * 0.8 + efficiency_bonus * 0.2

# 定義超參數空間
cifar_space = HyperparameterSpace(
    agent0_params=[],  # 這個例子沒有問題特定參數
    agent1_params=['conv1_filters', 'conv2_filters', 'hidden_size', 
                   'use_batch_norm', 'dropout_rate'],
    agent2_params=['batch_size', 'learning_rate', 'weight_decay'],
    bounds={
        # 架構參數
        'conv1_filters': (16, 64),
        'conv2_filters': (32, 128),
        'hidden_size': (64, 256),
        'use_batch_norm': (False, True),
        'dropout_rate': (0.0, 0.7),
        
        # 訓練參數
        'batch_size': (16, 128),
        'learning_rate': (1e-4, 1e-2),
        'weight_decay': (1e-5, 1e-3)
    },
    param_types={
        'conv1_filters': int,
        'conv2_filters': int,
        'hidden_size': int,
        'use_batch_norm': bool,
        'dropout_rate': float,
        'batch_size': int,
        'learning_rate': 'log_uniform',
        'weight_decay': 'log_uniform'
    }
)

# 運行優化
environment = CIFAR10Environment()
config = DefaultConfigs.standard()
config.max_steps = 50  # 由於是演示，使用較少步數

optimizer = MAT_HPO_Optimizer(environment, cifar_space, config)
results = optimizer.optimize()

print("🎉 CIFAR-10 優化完成!")
print(f"最佳準確率: {results['best_performance']['accuracy']:.4f}")
print(f"最佳超參數: {results['best_hyperparameters']}")
```

### 範例2: 自然語言處理優化

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class NLPEnvironment(BaseEnvironment):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__(name="NLP-Classification")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def load_data(self):
        # 使用GLUE數據集的子集
        dataset = load_dataset('glue', 'sst2')
        
        # 使用小子集進行快速實驗
        train_dataset = dataset['train'].select(range(1000))
        val_dataset = dataset['validation'].select(range(200))
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        return {"train_size": len(train_dataset), "val_size": len(val_dataset)}
    
    def create_model(self, hyperparams):
        from transformers import AutoModelForSequenceClassification
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            hidden_dropout_prob=hyperparams.get('dropout_rate', 0.1),
            attention_probs_dropout_prob=hyperparams.get('attention_dropout', 0.1)
        )
        
        # 凍結部分層 (可選)
        if hyperparams.get('freeze_layers', 0) > 0:
            layers_to_freeze = hyperparams['freeze_layers']
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False
            for layer in model.bert.encoder.layer[:layers_to_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        return model
    
    def train_evaluate(self, model, hyperparams):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 數據預處理
        def preprocess_data(examples):
            return self.tokenizer(examples['sentence'], 
                                 truncation=True, 
                                 padding='max_length',
                                 max_length=128)
        
        train_dataset = self.train_dataset.map(preprocess_data, batched=True)
        val_dataset = self.val_dataset.map(preprocess_data, batched=True)
        
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        
        # 數據加載器
        train_loader = DataLoader(train_dataset, 
                                 batch_size=hyperparams['batch_size'],
                                 shuffle=True)
        val_loader = DataLoader(val_dataset,
                               batch_size=hyperparams['batch_size'])
        
        # 優化器 (使用不同學習率為不同層)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': hyperparams.get('weight_decay', 0.01),
                'lr': hyperparams['learning_rate']
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': hyperparams['learning_rate']
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        
        # 學習率調度器
        total_steps = len(train_loader) * hyperparams.get('epochs', 3)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps
        )
        
        # 訓練循環
        model.train()
        for epoch in range(hyperparams.get('epochs', 3)):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                
                outputs = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels)
                
                loss = outputs.loss
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
        
        # 評估
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label']
                
                outputs = model(input_ids=input_ids,
                               attention_mask=attention_mask)
                
                preds = torch.argmax(outputs.logits, dim=-1).cpu()
                predictions.extend(preds.tolist())
                true_labels.extend(labels.tolist())
        
        # 計算指標
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'num_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    
    def compute_reward(self, metrics):
        f1 = metrics['f1']
        accuracy = metrics['accuracy']
        num_params = metrics.get('num_parameters', 100000)
        
        # 平衡效能和模型大小
        efficiency_score = max(0, 1.0 - num_params / 10000000)  # 參數越少越好
        
        return f1 * 0.5 + accuracy * 0.3 + efficiency_score * 0.2

# NLP超參數空間
nlp_space = HyperparameterSpace(
    agent0_params=['freeze_layers'],  # 領域特定：凍結層數
    agent1_params=['dropout_rate', 'attention_dropout'],  # 架構參數
    agent2_params=['batch_size', 'learning_rate', 'weight_decay'],  # 訓練參數
    bounds={
        'freeze_layers': (0, 6),  # BERT base有12層，凍結0-6層
        'dropout_rate': (0.0, 0.5),
        'attention_dropout': (0.0, 0.3),
        'batch_size': (8, 32),  # NLP任務通常使用較小批次
        'learning_rate': (1e-5, 1e-4),  # 預訓練模型需要較小學習率
        'weight_decay': (1e-3, 1e-1)
    },
    param_types={
        'freeze_layers': int,
        'dropout_rate': float,
        'attention_dropout': float,
        'batch_size': int,
        'learning_rate': 'log_uniform',
        'weight_decay': 'log_uniform'
    }
)
```

## 常見問題

### Q1: 如何選擇合適的代理分工？

**A:** 代理分工的原則是按參數性質和優化策略分組：

```python
# 好的分工範例
Agent 0: 問題領域參數 (類別權重、數據增強、正規化)
Agent 1: 模型架構參數 (層數、神經元數、激活函數)  
Agent 2: 訓練過程參數 (學習率、批次大小、優化器)

# 避免的分工
❌ 按字母順序分配
❌ 隨機分配  
❌ 所有參數都給一個代理
```

### Q2: 如何處理類別不平衡的參數類型？

**A:** 使用參數類型標記和自訂處理：

```python
# 對於對數分布的參數
'learning_rate': 'log_uniform'

# 對於類別參數
'optimizer': ['adam', 'sgd', 'rmsprop']  # 自動識別為類別

# 對於條件參數
def conditional_constraint(hyperparams):
    if hyperparams['optimizer'] == 'sgd':
        hyperparams['momentum'] = 0.9  # SGD需要動量
    else:
        hyperparams['momentum'] = 0.0  # 其他優化器不需要
    return hyperparams
```

### Q3: 訓練過程中記憶體不足怎麼辦？

**A:** 使用梯度累積和記憶體優化：

```python
def train_evaluate(self, model, hyperparams):
    # 梯度累積
    accumulation_steps = max(1, 64 // hyperparams['batch_size'])
    
    for i, batch in enumerate(train_loader):
        outputs = model(batch)
        loss = criterion(outputs, targets) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # 記憶體清理
    torch.cuda.empty_cache()
```

### Q4: 如何處理長時間運行的優化？

**A:** 使用檢查點和監控：

```python
# 啟用檢查點
environment = MyEnvironment(checkpoint_dir="./checkpoints")

# 添加監控回調
def monitoring_callback(env, hyperparams, metrics, reward):
    # 定期保存
    if env.current_step % 10 == 0:
        env._save_checkpoint()
    
    # 異常檢測
    if reward < -1.0:  # 異常低獎勵
        print("⚠️ 檢測到異常低獎勵，可能需要檢查")

environment.add_step_callback(monitoring_callback)
```

### Q5: 如何優化搜索效率？

**A:** 使用多種策略：

```python
# 1. 使用早停
config.early_stop_patience = 20

# 2. 適應性搜索空間
space = AdaptiveHyperparameterSpace(...)

# 3. 先驗知識初始化
space = HyperparameterSpace(
    ...,
    default_values={
        'learning_rate': 1e-3,  # 經驗值
        'batch_size': 32,
        'optimizer': 'adam'
    }
)

# 4. 分階段優化
# 第一階段：粗搜索
config_coarse = DefaultConfigs.quick_test()
results_coarse = optimizer.optimize()

# 第二階段：細搜索 (縮小搜索範圍)
refined_space = create_refined_space(results_coarse['best_hyperparameters'])
```

## 最佳實踐

### 1. 實驗設計

```python
# ✅ 好的實驗設計
class ExperimentManager:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = f"./experiments/{experiment_name}_{self.timestamp}"
        
    def setup_experiment(self):
        # 創建目錄結構
        os.makedirs(f"{self.base_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.base_dir}/results", exist_ok=True)
        os.makedirs(f"{self.base_dir}/logs", exist_ok=True)
        
        # 保存配置
        self.save_config()
        
        # 設置日誌
        self.setup_logging()
        
    def save_config(self):
        config = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'random_seed': 42
        }
        
        with open(f"{self.base_dir}/experiment_config.json", 'w') as f:
            json.dump(config, f, indent=2)

# 使用方式
experiment = ExperimentManager("cifar10_optimization")
experiment.setup_experiment()
```

### 2. 超參數空間設計

```python
# ✅ 結構化的空間定義
def create_comprehensive_space(problem_type='classification'):
    """根據問題類型創建合適的搜索空間"""
    
    if problem_type == 'classification':
        return HyperparameterSpace(
            agent0_params=[
                'class_weight_method',  # 類別平衡方法
                'data_augmentation',    # 數據增強強度
                'regularization_type'   # 正規化類型
            ],
            agent1_params=[
                'architecture_type',    # 架構類型
                'depth',               # 網路深度
                'width',               # 網路寬度
                'activation'           # 激活函數
            ],
            agent2_params=[
                'optimizer_type',       # 優化器
                'learning_rate',        # 學習率
                'batch_size',          # 批次大小
                'schedule_type'        # 學習率調度
            ],
            bounds=get_classification_bounds(),
            param_types=get_classification_types()
        )
    elif problem_type == 'regression':
        return create_regression_space()
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
```

### 3. 效能監控

```python
# ✅ 完整的監控系統
class PerformanceMonitor:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics_history = []
        
    def log_step(self, step, metrics, hyperparams):
        """記錄每步的詳細信息"""
        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'hyperparams': hyperparams,
            'system_info': self.get_system_info()
        }
        
        self.metrics_history.append(log_entry)
        
        # 保存到文件
        with open(f"{self.log_dir}/step_{step:04d}.json", 'w') as f:
            json.dump(log_entry, f, indent=2)
    
    def get_system_info(self):
        """獲取系統資源信息"""
        import psutil
        
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_memory': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
    
    def generate_report(self):
        """生成優化報告"""
        if not self.metrics_history:
            return
        
        # 效能趨勢分析
        rewards = [entry['metrics'].get('reward', 0) for entry in self.metrics_history]
        best_idx = np.argmax(rewards)
        best_entry = self.metrics_history[best_idx]
        
        report = {
            'summary': {
                'total_steps': len(self.metrics_history),
                'best_step': best_idx,
                'best_reward': rewards[best_idx],
                'best_hyperparams': best_entry['hyperparams']
            },
            'convergence': {
                'final_reward': rewards[-1],
                'improvement': rewards[-1] - rewards[0],
                'plateau_detection': self.detect_plateau(rewards)
            }
        }
        
        with open(f"{self.log_dir}/optimization_report.json", 'w') as f:
            json.dump(report, f, indent=2)
```

### 4. 可重現性保證

```python
# ✅ 確保實驗可重現
def ensure_reproducibility(seed=42):
    """設置所有隨機種子以確保可重現性"""
    import random
    import numpy as np
    import torch
    
    # Python隨機種子
    random.seed(seed)
    
    # NumPy隨機種子
    np.random.seed(seed)
    
    # PyTorch隨機種子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 確保CUDNN的確定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 設置環境變量
    os.environ['PYTHONHASHSEED'] = str(seed)

# 在每個實驗開始前調用
ensure_reproducibility(42)
```

### 5. 錯誤處理和恢復

```python
# ✅ 健壯的錯誤處理
class RobustOptimizer:
    def __init__(self, environment, space, config):
        self.environment = environment
        self.space = space  
        self.config = config
        self.failed_configs = []
        
    def optimize_with_retry(self, max_retries=3):
        """帶重試機制的優化"""
        for attempt in range(max_retries):
            try:
                optimizer = MAT_HPO_Optimizer(self.environment, self.space, self.config)
                results = optimizer.optimize()
                return results
                
            except torch.cuda.OutOfMemoryError:
                print(f"GPU記憶體不足，嘗試減少批次大小 (嘗試 {attempt + 1}/{max_retries})")
                self._reduce_batch_size()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"優化過程發生錯誤: {e}")
                if attempt == max_retries - 1:
                    raise
                
                # 記錄失敗配置
                self.failed_configs.append({
                    'attempt': attempt,
                    'error': str(e),
                    'config': self.config.to_dict()
                })
                
        raise RuntimeError(f"優化在 {max_retries} 次嘗試後仍然失敗")
    
    def _reduce_batch_size(self):
        """減少批次大小以節省記憶體"""
        current_batch_bounds = self.space.bounds.get('batch_size', (8, 128))
        new_max = max(8, current_batch_bounds[1] // 2)
        self.space.bounds['batch_size'] = (current_batch_bounds[0], new_max)
        print(f"將批次大小上限調整為 {new_max}")
```

---

這個詳細使用說明涵蓋了MAT-HPO函式庫的所有重要功能和使用場景。無論您是初學者還是高級用戶，都能在這裡找到適合的指導和範例。

如果您有任何問題或需要更具體的幫助，請隨時詢問！