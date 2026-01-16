# MAT-HPO Library - Troubleshooting Guide

完整的問題診斷和解決方案指南。

## 目錄

- [常見問題快速修復](#-常見問題快速修復)
- [安裝問題](#-安裝問題)
- [運行時錯誤](#-運行時錯誤)
- [性能問題](#-性能問題)
- [LLM 相關問題](#-llm-相關問題)
- [記憶體問題](#-記憶體問題)
- [配置問題](#-配置問題)
- [調試技巧](#-調試技巧)

## 常見問題快速修復

| 問題 | 症狀 | 解決方案 | 命令 |
|------|------|----------|------|
| **❌ 導入錯誤** | `ModuleNotFoundError: No module named 'MAT_HPO_LIB'` | 設置 Python 路徑 | `export PYTHONPATH=$PYTHONPATH:$(pwd)` |
| **🚫 CUDA 記憶體不足** | `RuntimeError: CUDA out of memory` | 使用 CPU 模式 | `config = DefaultConfigs.cpu_only()` |
| **🔍 測試安裝** | 不確定是否安裝正確 | 運行功能測試 | `python test_working_examples.py` |
| **📦 缺少依賴** | `ImportError: No module named 'torch'` | 安裝依賴 | `pip install torch numpy scikit-learn` |
| **🐍 Python 版本** | `SyntaxError` 或版本錯誤 | 檢查 Python 版本 | `python --version` (需要 3.8+) |

## 安裝問題

### 問題 1: ModuleNotFoundError

**錯誤訊息**:
```
ModuleNotFoundError: No module named 'MAT_HPO_LIB'
```

**解決方案**:

1. **檢查當前目錄**:
```bash
pwd
# 應該在 MAT_HPO_Library 目錄中
```

2. **設置 Python 路徑**:
```bash
# 方法 1: 環境變數
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 方法 2: 在 Python 中設置
import sys
sys.path.append('/path/to/MAT_HPO_Library')
```

3. **驗證安裝**:
```bash
python -c "from MAT_HPO_LIB import EasyHPO; print('✅ 安裝成功')"
```

### 問題 2: 依賴安裝失敗

**錯誤訊息**:
```
ERROR: Could not find a version that satisfies the requirement torch
```

**解決方案**:

1. **更新 pip**:
```bash
pip install --upgrade pip
```

2. **安裝 PyTorch**:
```bash
# CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU 版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **安裝其他依賴**:
```bash
pip install numpy scikit-learn pandas matplotlib seaborn
```

### 問題 3: 權限錯誤

**錯誤訊息**:
```
PermissionError: [Errno 13] Permission denied
```

**解決方案**:

1. **使用用戶安裝**:
```bash
pip install --user torch numpy scikit-learn
```

2. **創建虛擬環境**:
```bash
python -m venv mat_hpo_env
source mat_hpo_env/bin/activate  # Linux/Mac
# 或
mat_hpo_env\Scripts\activate     # Windows
pip install torch numpy scikit-learn
```

## 🚨 運行時錯誤

### 問題 1: CUDA 相關錯誤

**錯誤訊息**:
```
RuntimeError: CUDA out of memory
RuntimeError: CUDA error: no kernel image is available for execution
```

**解決方案**:

1. **檢查 CUDA 可用性**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

2. **使用 CPU 模式**:
```python
from MAT_HPO_LIB.utils import DefaultConfigs
config = DefaultConfigs.cpu_only()
```

3. **減少批次大小**:
```python
config = OptimizationConfig(
    max_steps=50,
    batch_size=8,  # 減少批次大小
    use_cuda=False  # 強制使用 CPU
)
```

### 問題 2: 數據格式錯誤

**錯誤訊息**:
```
ValueError: Expected 2D array, got 1D array instead
TypeError: 'NoneType' object is not iterable
```

**解決方案**:

1. **檢查數據格式**:
```python
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_train type: {type(X_train)}")
print(f"y_train type: {type(y_train)}")
```

2. **確保數據正確格式**:
```python
import numpy as np

# 確保是 numpy 數組
X_train = np.array(X_train)
y_train = np.array(y_train)

# 檢查維度
if len(X_train.shape) == 1:
    X_train = X_train.reshape(-1, 1)
```

3. **檢查數據完整性**:
```python
# 檢查是否有 NaN 值
print(f"X_train NaN count: {np.isnan(X_train).sum()}")
print(f"y_train NaN count: {np.isnan(y_train).sum()}")

# 檢查是否有無限值
print(f"X_train inf count: {np.isinf(X_train).sum()}")
```

### 問題 3: 超參數空間錯誤

**錯誤訊息**:
```
ValueError: Invalid hyperparameter space configuration
KeyError: 'learning_rate'
```

**解決方案**:

1. **檢查超參數空間定義**:
```python
from MAT_HPO_LIB.core.hyperparameter_space import HyperparameterSpace

space = HyperparameterSpace()

# 正確的定義方式
space.add_continuous('learning_rate', min_val=1e-5, max_val=1e-1, agent=0)
space.add_discrete('batch_size', choices=[16, 32, 64, 128], agent=1)
space.add_boolean('use_dropout', agent=2)

# 檢查空間
print(f"Space parameters: {space.get_parameter_names()}")
```

2. **確保參數名稱一致**:
```python
# 在環境中使用相同的參數名稱
def create_model(self, hyperparams):
    return MyModel(
        learning_rate=hyperparams['learning_rate'],  # 必須與空間定義一致
        batch_size=hyperparams['batch_size'],
        use_dropout=hyperparams['use_dropout']
    )
```

## 性能問題

### 問題 1: 優化速度太慢

**症狀**: 每個步驟耗時過長

**解決方案**:

1. **減少優化步數**:
```python
config = OptimizationConfig(
    max_steps=20,  # 減少步數
    batch_size=16,  # 減少批次大小
    behaviour_update_freq=5  # 減少更新頻率
)
```

2. **使用快速測試配置**:
```python
from MAT_HPO_LIB.utils import DefaultConfigs
config = DefaultConfigs.quick_test()  # 只有 10 步
```

3. **禁用詳細日誌**:
```python
config = OptimizationConfig(
    max_steps=50,
    verbose=False,  # 禁用詳細輸出
    log_level="WARNING"
)
```

### 問題 2: 記憶體使用過高

**症狀**: 系統記憶體不足或 GPU 記憶體不足

**解決方案**:

1. **監控記憶體使用**:
```python
import psutil
import torch

# CPU 記憶體
memory = psutil.virtual_memory()
print(f"CPU Memory: {memory.percent}% used")

# GPU 記憶體
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"GPU Memory Max: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
```

2. **優化記憶體使用**:
```python
config = OptimizationConfig(
    max_steps=30,
    batch_size=8,  # 減少批次大小
    use_cuda=False,  # 使用 CPU
    gradient_clip=0.5  # 梯度裁剪
)
```

3. **清理記憶體**:
```python
import gc
import torch

# 清理 Python 記憶體
gc.collect()

# 清理 GPU 記憶體
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

## 🤖 LLM 相關問題

### 問題 1: LLM 連接失敗

**錯誤訊息**:
```
ConnectionError: Failed to connect to LLM service
TimeoutError: LLM request timed out
```

**解決方案**:

1. **檢查 Ollama 服務**:
```bash
# 檢查 Ollama 是否運行
curl http://localhost:11434/api/tags

# 啟動 Ollama 服務
ollama serve
```

2. **檢查模型可用性**:
```bash
# 列出可用模型
ollama list

# 拉取所需模型
ollama pull llama3.2:3b
```

3. **使用備用 LLM 配置**:
```python
optimizer = EasyHPO(
    task_type="classification",
    llm_enabled=True,
    llm_model="llama3.2:1b",  # 使用較小的模型
    llm_timeout=30  # 增加超時時間
)
```

### 問題 2: LLM 回應格式錯誤

**錯誤訊息**:
```
ValueError: Invalid LLM response format
JSONDecodeError: Expecting value
```

**解決方案**:

1. **檢查 LLM 回應**:
```python
from MAT_HPO_LIB.llm import OllamaLLMClient

client = OllamaLLMClient("llama3.2:3b")
response = client.generate_response("Test prompt")
print(f"LLM Response: {response}")
```

2. **使用更穩定的模型**:
```python
optimizer = EasyHPO(
    task_type="classification",
    llm_enabled=True,
    llm_model="llama3.2:1b",  # 更穩定的模型
    llm_strategy="fixed_alpha"  # 使用固定策略
)
```

3. **禁用 LLM 進行測試**:
```python
optimizer = EasyHPO(
    task_type="classification",
    llm_enabled=False  # 暫時禁用 LLM
)
```

## 💾 記憶體問題

### 問題 1: 記憶體洩漏

**症狀**: 記憶體使用持續增長

**解決方案**:

1. **定期清理記憶體**:
```python
import gc
import torch

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 在優化過程中定期調用
for step in range(max_steps):
    # 優化步驟
    optimizer.step()
    
    # 每 10 步清理一次
    if step % 10 == 0:
        cleanup_memory()
```

2. **使用記憶體監控**:
```python
from MAT_HPO_LIB.utils import MemoryMonitor

monitor = MemoryMonitor()
config = OptimizationConfig(
    max_steps=100,
    memory_monitor=monitor
)
```

### 問題 2: 大數據集處理

**症狀**: 數據集太大無法載入

**解決方案**:

1. **使用數據生成器**:
```python
class DataGenerator:
    def __init__(self, X, y, batch_size=1000):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.current_idx = 0
    
    def get_batch(self):
        start = self.current_idx
        end = min(start + self.batch_size, len(self.X))
        batch_X = self.X[start:end]
        batch_y = self.y[start:end]
        self.current_idx = end
        return batch_X, batch_y
```

2. **分批處理**:
```python
def process_large_dataset(X, y, batch_size=1000):
    results = []
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        # 處理批次
        result = process_batch(batch_X, batch_y)
        results.append(result)
    return results
```

## ⚙️ 配置問題

### 問題 1: 配置參數無效

**錯誤訊息**:
```
ValueError: Invalid configuration parameter
TypeError: Unexpected keyword argument
```

**解決方案**:

1. **檢查配置參數**:
```python
from MAT_HPO_LIB.utils import OptimizationConfig

# 正確的配置
config = OptimizationConfig(
    max_steps=100,        # 整數
    batch_size=32,        # 整數
    use_cuda=True,        # 布林值
    learning_rate=1e-3,   # 浮點數
    exploration_rate=0.1  # 0-1 之間的浮點數
)

# 檢查配置
print(f"Config: {config}")
```

2. **使用預設配置**:
```python
from MAT_HPO_LIB.utils import DefaultConfigs

# 使用預設配置
config = DefaultConfigs.standard()
```

### 問題 2: 環境配置錯誤

**錯誤訊息**:
```
AttributeError: 'MyEnvironment' object has no attribute 'load_data'
NotImplementedError: Abstract method not implemented
```

**解決方案**:

1. **檢查環境實現**:
```python
from MAT_HPO_LIB import BaseEnvironment

class MyEnvironment(BaseEnvironment):
    def load_data(self):
        # 必須實現
        return {'train_X': X_train, 'train_Y': y_train}
    
    def create_model(self, hyperparams):
        # 必須實現
        return MyModel(**hyperparams)
    
    def train_evaluate(self, model, hyperparams):
        # 必須實現
        return {'f1': 0.85}
    
    def compute_reward(self, metrics):
        # 必須實現
        return metrics['f1']
```

2. **使用預建環境**:
```python
from MAT_HPO_LIB.core.enhanced_environment import TimeSeriesEnvironment

# 使用預建的環境
environment = TimeSeriesEnvironment("MyDataset")
```

## 🔍 調試技巧

### 1. 啟用詳細日誌

```python
import logging

# 設置日誌級別
logging.basicConfig(level=logging.DEBUG)

# 在配置中啟用詳細輸出
config = OptimizationConfig(
    max_steps=10,
    verbose=True,
    log_level="DEBUG"
)
```

### 2. 使用測試模式

```python
# 使用快速測試配置
config = DefaultConfigs.quick_test()

# 減少數據大小進行測試
X_train_small = X_train[:100]
y_train_small = y_train[:100]
```

### 3. 檢查中間結果

```python
# 在環境中添加調試輸出
def train_evaluate(self, model, hyperparams):
    print(f"Training with hyperparams: {hyperparams}")
    
    # 訓練邏輯
    metrics = train_and_evaluate(model, hyperparams)
    
    print(f"Training completed. Metrics: {metrics}")
    return metrics
```

### 4. 使用斷點調試

```python
import pdb

def train_evaluate(self, model, hyperparams):
    pdb.set_trace()  # 設置斷點
    # 在這裡可以檢查變數狀態
    metrics = train_and_evaluate(model, hyperparams)
    return metrics
```

## 📞 獲取幫助

如果以上解決方案都無法解決您的問題：

1. **檢查 GitHub Issues**: [MAT-HPO-Library Issues](https://github.com/VM230705/MAT-HPO-Library/issues)
2. **創建新的 Issue**: 提供詳細的錯誤訊息和環境信息
3. **查看文檔**: [完整文檔](https://vm230705.github.io/MAT-HPO-Library/)
4. **社區討論**: [GitHub Discussions](https://github.com/VM230705/MAT-HPO-Library/discussions)

### 報告問題時請包含：

- Python 版本
- 操作系統
- 錯誤的完整堆疊追蹤
- 使用的配置參數
- 數據集大小和類型
- 是否使用 GPU

---

**記住**: 大多數問題都可以通過使用預設配置和減少數據大小來快速診斷和解決！ 🚀