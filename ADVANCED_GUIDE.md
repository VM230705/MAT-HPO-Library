# MAT-HPO Library - Advanced Guide

Complete guide for advanced usage, API reference, and production deployment.

## Table of Contents

- [User Interface Levels](#-user-interface-levels)
- [LLM Strategies](#-llm-strategies)
- [API Reference](#-api-reference)
- [Production Configuration](#-production-configuration)
- [Custom Environments](#-custom-environments)
- [Performance Tuning](#-performance-tuning)
- [Integration Examples](#-integration-examples)

## User Interface Levels

MAT-HPO provides three levels of interfaces to accommodate different needs:

### Level 1: EasyHPO (Recommended for Most Users)

**Best for**: Quick start, prototyping, standard ML tasks

```python
from MAT_HPO_LIB import EasyHPO

# One-liner optimization
optimizer = EasyHPO(task_type="time_series_classification", max_trials=30)
results = optimizer.optimize(X_train, y_train, X_val, y_val)
```

**Features**:
- **Zero boilerplate** - just pass your data
- **Automatic model creation** for common tasks
- **Built-in training loops** and evaluation
- **Smart defaults** based on task type
- **LLM enhancement** with simple flags
- **Auto dataset analysis** and feature detection

### Level 2: FullControlHPO (Production Interface)

**Best for**: Production environments, custom pipelines with smart defaults

```python
from MAT_HPO_LIB import FullControlHPO

optimizer = FullControlHPO(
    task_type="time_series_classification",
    max_trials=30,
    llm_enabled=True,
    llm_strategy="adaptive",
    alpha=0.3,
    # ICBEB-optimized RL parameters
    replay_buffer_size=150,
    batch_size=12,
    actor_lr=0.0008,
    critic_lr=0.0015
)
results = optimizer.optimize(X_train, y_train, X_val, y_val)
```

**Features**:
- **Pre-built environments** for time series, ECG, etc.
- **Configuration presets** for common scenarios
- **Automatic dataset analysis** with LLM context
- **Flexible customization** while maintaining smart defaults
- **Multi-fidelity optimization** support

### Level 3: Core Components (Full Control)

**Best for**: Research, custom algorithms, maximum flexibility

```python
from MAT_HPO_LIB import MAT_HPO_Optimizer, BaseEnvironment, HyperparameterSpace

class MyEnvironment(BaseEnvironment):
    def load_data(self): # Your implementation
    def create_model(self, params): # Your implementation
    def train_evaluate(self, model, params): # Your implementation
    def compute_reward(self, metrics): # Your implementation

# Full manual control
optimizer = MAT_HPO_Optimizer(environment, space, config)
```

**Features**:
- **Complete control** over every aspect
- **Research-grade flexibility** for novel algorithms
- **Multi-agent RL** with SQDDPG algorithm
- **Custom reward functions** and metrics
- **Advanced configuration** options

## LLM Strategies

Based on paper [arXiv:2507.13712](https://arxiv.org/abs/2507.13712), MAT-HPO supports intelligent hyperparameter suggestions.

### 1. Fixed Alpha Strategy

**Description**: Use fixed mixing ratio combining LLM and RL suggestions.

**Use Cases**:
- New users wanting simple, stable results
- Known optimal mixing ratios
- Production environments requiring predictable behavior

```python
optimizer = EasyHPO(
    task_type="time_series_classification",
    llm_enabled=True,
    llm_strategy="fixed_alpha",
    alpha=0.3  # 30% LLM + 70% RL
)
```

### 2. Adaptive Strategy (Recommended)

**Description**: Monitor RL performance improvement slope, trigger LLM when slope < threshold.

**Use Cases**:
- Intelligent automatic adjustment
- Uncertain optimal intervention timing
- Research and experimental scenarios
- Production environments pursuing optimal performance

```python
optimizer = EasyHPO(
    task_type="time_series_classification",
    llm_enabled=True,
    llm_strategy="adaptive",
    slope_threshold=0.01  # Trigger LLM when RL slope < 0.01
)
```

### LLM Configuration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `llm_model` | LLM model name | `"llama3.2:3b"` | Any Ollama model |
| `alpha` | Mixing ratio (fixed strategy) | `0.3` | 0.0-1.0 |
| `slope_threshold` | Performance slope threshold (adaptive) | `0.01` | 0.001-0.1 |
| `min_episodes` | Minimum episodes before LLM | `5` | 3-10 |
| `cooldown` | LLM intervention cooldown | `5` | 2-10 |

## 📚 API Reference

### HyperparameterSpace

Define your search space with different parameter types:

```python
space = HyperparameterSpace()

# 📈 Continuous parameters (float values)
space.add_continuous('learning_rate', min_val=1e-5, max_val=1e-1, agent=0)
space.add_continuous('dropout', min_val=0.0, max_val=0.8, agent=2)

# 🎯 Discrete parameters (specific choices)
space.add_discrete('batch_size', choices=[16, 32, 64, 128], agent=1)
space.add_discrete('optimizer', choices=['adam', 'sgd', 'rmsprop'], agent=2)

# ✅ Boolean parameters (True/False)
space.add_boolean('use_batch_norm', agent=1)
space.add_boolean('use_dropout', agent=1)
```

### BaseEnvironment

Create your custom optimization environment:

```python
class YourEnvironment(BaseEnvironment):
    def __init__(self, dataset_name="MyDataset"):
        super().__init__(dataset_name)
        # Initialize your data, models, etc.

    def load_data(self):
        """Load and prepare your dataset"""
        return {
            'train_X': X_train, 'train_Y': y_train,
            'val_X': X_val, 'val_Y': y_val,
            'test_X': X_test, 'test_Y': y_test
        }

    def create_model(self, hyperparams):
        """Create model with given hyperparameters"""
        return YourModel(**hyperparams)

    def train_evaluate(self, model, hyperparams):
        """Train and evaluate model"""
        # Your training logic here
        metrics = train_and_evaluate(model, hyperparams)
        return metrics

    def compute_reward(self, metrics):
        """Compute reward from metrics"""
        return metrics.get('f1', 0.0)
```

### OptimizationConfig

Configure optimization parameters:

```python
from MAT_HPO_LIB.utils import OptimizationConfig

config = OptimizationConfig(
    max_steps=100,           # Number of optimization steps
    batch_size=32,           # Training batch size
    use_cuda=True,           # Enable GPU acceleration
    policy_learning_rate=1e-3,  # Policy network learning rate
    noise_std=0.1,             # Exploration noise standard deviation
    gradient_clip=1.0,       # Gradient clipping
    verbose=True             # Enable detailed logging
)
```

## 🏭 Production Configuration

### SPNV2 ECG Classification Example

```python
from MAT_HPO_LIB import FullControlHPO

# Production configuration for ECG classification
optimizer = FullControlHPO(
    task_type="time_series_classification",
    max_trials=30,
    # LLM Configuration
    llm_enabled=True,
    llm_model="llama3.2:3b",
    llm_strategy="adaptive",
    slope_threshold=0.01,
    # ICBEB-optimized RL parameters
    replay_buffer_size=150,
    batch_size=12,
    actor_lr=0.0008,
    critic_lr=0.0015,
    # Performance optimization
    gradient_clip=1.0,
    behaviour_update_freq=1,
    target_update_freq=5
)

results = optimizer.optimize(X_train, y_train, X_val, y_val)
```

### Dataset-Specific Configurations

#### ICBEB Dataset
```python
# Optimized for ICBEB ECG classification
config = {
    'replay_buffer_size': 150,
    'batch_size': 12,
    'actor_lr': 0.0008,
    'critic_lr': 0.0015,
    'gradient_clip': 1.0,
    'behaviour_update_freq': 1
}
```

#### PTBXL Dataset
```python
# Optimized for PTBXL ECG classification
config = {
    'replay_buffer_size': 200,
    'batch_size': 16,
    'actor_lr': 0.001,
    'critic_lr': 0.002,
    'gradient_clip': 0.5,
    'behaviour_update_freq': 2
}
```

## 🛠️ Custom Environments

### Time Series Classification Environment

```python
from MAT_HPO_LIB.core.enhanced_environment import TimeSeriesEnvironment

class ECGEnvironment(TimeSeriesEnvironment):
    def __init__(self, dataset_name="ECG"):
        super().__init__(dataset_name)
        self.num_classes = 9  # ICBEB has 9 classes
        
    def create_model(self, hyperparams):
        """Create LSTM model for ECG classification"""
        return LSTMClassifier(
            input_size=hyperparams.get('input_size', 1000),
            hidden_size=hyperparams.get('hidden_size', 128),
            num_classes=self.num_classes,
            dropout=hyperparams.get('dropout', 0.2)
        )
    
    def train_evaluate(self, model, hyperparams):
        """Train and evaluate the model"""
        # Your training logic
        train_loss = train_model(model, self.train_data, hyperparams)
        val_metrics = evaluate_model(model, self.val_data)
        return val_metrics
```

### Custom Reward Functions

```python
def custom_reward_function(metrics):
    """Custom reward combining multiple metrics"""
    f1 = metrics.get('f1', 0.0)
    auc = metrics.get('auc', 0.0)
    gmean = metrics.get('gmean', 0.0)
    
    # Weighted combination
    reward = 0.5 * f1 + 0.3 * auc + 0.2 * gmean
    return reward

# Use in environment
class MyEnvironment(BaseEnvironment):
    def compute_reward(self, metrics):
        return custom_reward_function(metrics)
```

## Performance Tuning

### Memory Optimization

```python
# For large datasets or limited GPU memory
config = OptimizationConfig(
    max_steps=50,
    batch_size=16,           # Smaller batch size
    use_cuda=True,
    gradient_clip=0.5,       # More aggressive clipping
    behaviour_update_freq=5  # Less frequent updates
)
```

### CPU-Only Mode

```python
# For environments without GPU
config = DefaultConfigs.cpu_only()
optimizer = MAT_HPO_Optimizer(environment, space, config)
```

### Multi-Fidelity Optimization

```python
from MAT_HPO_LIB import MultiFidelityMAT_HPO, FidelityConfig

# Define fidelity levels
fidelity_config = FidelityConfig(
    levels=[0.1, 0.3, 0.6, 1.0],  # 10%, 30%, 60%, 100% of data
    costs=[1, 3, 6, 10]           # Relative computational costs
)

optimizer = MultiFidelityMAT_HPO(
    environment, space, config, fidelity_config
)
```

## 🔗 Integration Examples

### SPNV2 Integration

```python
# In SPNV2/2. SPL_HPO_Enhanced.py
from MAT_HPO_LIB import EasyHPO

def run_spnv2_hpo(dataset, fold, steps, use_llm=True):
    optimizer = EasyHPO(
        task_type="ecg_classification",
        llm_enabled=use_llm,
        llm_strategy="adaptive",
        max_trials=steps
    )
    
    # Load SPNV2 data
    X_train, y_train, X_val, y_val = load_spnv2_data(dataset, fold)
    
    # Run optimization
    results = optimizer.optimize(X_train, y_train, X_val, y_val)
    
    return results
```

### PyTorch Integration

```python
import torch
from MAT_HPO_LIB import MAT_HPO_Optimizer, BaseEnvironment

class PyTorchEnvironment(BaseEnvironment):
    def create_model(self, hyperparams):
        return torch.nn.Sequential(
            torch.nn.Linear(hyperparams['input_size'], hyperparams['hidden_size']),
            torch.nn.ReLU(),
            torch.nn.Dropout(hyperparams['dropout']),
            torch.nn.Linear(hyperparams['hidden_size'], hyperparams['num_classes'])
        )
    
    def train_evaluate(self, model, hyperparams):
        # PyTorch training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training logic here
        # ...
        
        return {'f1': f1_score, 'accuracy': accuracy}
```

### Scikit-learn Integration

```python
from sklearn.ensemble import RandomForestClassifier
from MAT_HPO_LIB import EasyHPO

# EasyHPO automatically handles scikit-learn models
optimizer = EasyHPO(
    task_type="classification",
    model_type="random_forest",
    max_trials=20
)

results = optimizer.optimize(X_train, y_train, X_val, y_val)
```

## Monitoring and Logging

### WandB Integration

```python
import wandb
from MAT_HPO_LIB import MAT_HPO_Optimizer

# Initialize WandB
wandb.init(project="my-hpo-project")

class WandBEnvironment(BaseEnvironment):
    def train_evaluate(self, model, hyperparams):
        # Your training logic
        metrics = train_and_evaluate(model, hyperparams)
        
        # Log to WandB
        wandb.log({
            'f1': metrics['f1'],
            'accuracy': metrics['accuracy'],
            'learning_rate': hyperparams['learning_rate'],
            'batch_size': hyperparams['batch_size']
        })
        
        return metrics
```

### Custom Logging

```python
from MAT_HPO_LIB.utils.logger import HPOLogger

# Create custom logger
logger = HPOLogger(
    log_dir="./logs",
    log_level="INFO",
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Use in optimization
config = OptimizationConfig(
    max_steps=100,
    logger=logger,
    verbose=True
)
```

## Best Practices

### 1. Hyperparameter Space Design

```python
# Good: Balanced agent assignment
space = HyperparameterSpace()
space.add_continuous('learning_rate', 1e-5, 1e-2, agent=2)  # Training
space.add_discrete('hidden_size', [64, 128, 256, 512], agent=1)  # Architecture
space.add_continuous('class_weight_0', 0.5, 2.0, agent=0)  # Problem-specific

# Avoid: All parameters assigned to one agent
space.add_continuous('learning_rate', 1e-5, 1e-2, agent=0)
space.add_continuous('batch_size', 16, 128, agent=0)  # Bad!
```

### 2. Reward Function Design

```python
# Good: Balanced and interpretable
def balanced_reward(metrics):
    f1 = metrics.get('f1', 0.0)
    auc = metrics.get('auc', 0.0)
    return 0.7 * f1 + 0.3 * auc

# Avoid: Overly complex or unstable
def complex_reward(metrics):
    # Too many terms can make optimization unstable
    return (f1 * auc * gmean) / (1 + abs(f1 - auc))  # Bad!
```

### 3. Environment Design

```python
# Good: Clear separation of concerns
class MyEnvironment(BaseEnvironment):
    def load_data(self):
        # Only data loading
        pass
    
    def create_model(self, hyperparams):
        # Only model creation
        pass
    
    def train_evaluate(self, model, hyperparams):
        # Only training and evaluation
        pass

# Avoid: Mixing concerns
class BadEnvironment(BaseEnvironment):
    def train_evaluate(self, model, hyperparams):
        # Don't load data here!
        data = self.load_data()  # Bad!
        # Don't create model here!
        model = self.create_model(hyperparams)  # Bad!
```

## Advanced Configuration

### Custom LLM Clients

```python
from MAT_HPO_LIB.llm import BaseLLMClient

class CustomLLMClient(BaseLLMClient):
    def __init__(self, api_key, model_name):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
    
    def generate_response(self, prompt, **kwargs):
        # Your custom LLM integration
        response = call_your_llm_api(prompt, self.api_key, self.model_name)
        return self.parse_response(response)

# Use custom client
optimizer = EasyHPO(
    task_type="classification",
    llm_enabled=True,
    llm_client=CustomLLMClient("your-api-key", "your-model")
)
```

### Custom Metrics

```python
from MAT_HPO_LIB.utils.metrics import MetricsCalculator

class CustomMetricsCalculator(MetricsCalculator):
    def calculate_basic_metrics(self, y_true, y_pred, y_proba=None):
        metrics = super().calculate_basic_metrics(y_true, y_pred, y_proba)
        
        # Add custom metrics
        metrics['custom_metric'] = self.calculate_custom_metric(y_true, y_pred)
        
        return metrics
    
    def calculate_custom_metric(self, y_true, y_pred):
        # Your custom metric calculation
        return custom_metric_value
```

This advanced guide covers all the key aspects of using MAT-HPO Library in production environments. For troubleshooting and common issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
