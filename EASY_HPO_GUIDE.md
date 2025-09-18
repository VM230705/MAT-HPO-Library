# EasyHPO Interface Guide

The EasyHPO interface provides a simplified, user-friendly way to perform hyperparameter optimization with MAT_HPO_LIB. It's designed to require minimal setup while providing powerful optimization capabilities.

## Quick Start

### 1. Basic Usage - Just Pass Data

The simplest way to use EasyHPO is to just pass your data:

```python
from MAT_HPO_LIB import EasyHPO
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create your dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# One-liner optimization
optimizer = EasyHPO(task_type="classification", max_trials=30)
results = optimizer.optimize(X_train, y_train, X_val, y_val)

print(f"Best hyperparameters: {results['best_hyperparameters']}")
print(f"Best performance: {results['best_performance']}")
```

### 2. One-Liner Functions

For even simpler usage, use the convenience functions:

```python
from MAT_HPO_LIB.easy_hpo import quick_optimize

# Ultra-simple one-liner
best_params = quick_optimize(X_train, y_train, X_val, y_val,
                            task_type="classification", max_trials=20)
print(f"Best params: {best_params}")
```

## Task Types and Auto-Configuration

EasyHPO automatically configures hyperparameter spaces based on task type:

### Time Series Classification
```python
optimizer = EasyHPO(task_type="time_series_classification", max_trials=30)
results = optimizer.optimize(X_train, y_train, X_val, y_val)
```

**Automatically optimizes:**
- `hidden_size`: [32, 64, 128, 256, 512]
- `learning_rate`: [0.0001, 0.01]
- `batch_size`: [16, 32, 64, 128]
- `dropout`: [0.0, 0.5]
- `class_weight_N`: [0.5, 3.0] (for imbalanced datasets)

### ECG Classification
```python
optimizer = EasyHPO(task_type="ecg_classification", max_trials=30)
results = optimizer.optimize(X_train, y_train, X_val, y_val)
```

**Or use the specialized function:**
```python
from MAT_HPO_LIB.easy_hpo import quick_ecg_optimize

best_params = quick_ecg_optimize(X_train, y_train, X_val, y_val)
```

### Generic Classification
```python
optimizer = EasyHPO(task_type="classification", max_trials=30)
results = optimizer.optimize(X_train, y_train, X_val, y_val)
```

## LLM Enhancement

Enable LLM-enhanced optimization for intelligent hyperparameter suggestions:

```python
optimizer = EasyHPO(
    task_type="time_series_classification",
    llm_enabled=True,
    llm_model="llama3.2:3b",
    llm_strategy="adaptive",  # "fixed_alpha", "adaptive", "performance_based"
    max_trials=30
)
results = optimizer.optimize(X_train, y_train, X_val, y_val)
```

**LLM Strategies:**
- `"fixed_alpha"`: Fixed mixing ratio between LLM and RL suggestions
- `"adaptive"`: Dynamically adjust based on performance trends
- `"performance_based"`: Use LLM when performance stagnates

## Advanced Usage - Custom Pipeline

For custom model creation and training logic:

```python
def my_model_factory(**hyperparams):
    """Create your custom model"""
    hidden_size = hyperparams.get('hidden_size', 64)
    # ... create and return your model
    return model

def my_trainer(model, X_train, y_train, **hyperparams):
    """Train your model"""
    # ... custom training logic
    return trained_model

def my_evaluator(model, X_val, y_val):
    """Evaluate your model"""
    # ... custom evaluation logic
    return {'accuracy': acc, 'f1': f1}

# Use with custom pipeline
optimizer = EasyHPO(task_type="time_series_classification")
results = optimizer.optimize_with_pipeline(
    data_loader=lambda: (X_train, y_train, X_val, y_val),
    model_factory=my_model_factory,
    trainer=my_trainer,
    evaluator=my_evaluator
)
```

## Configuration Options

### Basic Options
```python
optimizer = EasyHPO(
    task_type="time_series_classification",  # Task type for auto-config
    llm_enabled=True,                        # Enable LLM enhancement
    llm_model="llama3.2:3b",                 # LLM model to use
    llm_strategy="adaptive",                 # LLM mixing strategy
    max_trials=30,                           # Number of optimization trials
    timeout_minutes=120,                     # Maximum time (None for no limit)
    auto_save=True,                          # Auto-save results
    output_dir="./my_hpo_results",           # Output directory
    verbose=True                             # Print progress
)
```

### Custom Hyperparameter Space
```python
from MAT_HPO_LIB import HyperparameterSpace

# Define custom space
custom_space = HyperparameterSpace()
custom_space.add_continuous('learning_rate', 0.001, 0.1, agent=0)
custom_space.add_discrete('batch_size', [16, 32, 64], agent=1)
custom_space.add_continuous('dropout', 0.0, 0.5, agent=2)

# Use custom space
results = optimizer.optimize(X_train, y_train, X_val, y_val,
                           custom_space=custom_space)
```

## Results and Model Creation

### Accessing Results
```python
results = optimizer.optimize(X_train, y_train, X_val, y_val)

# Best hyperparameters
best_params = results['best_hyperparameters']

# Performance metrics
performance = results['best_performance']
print(f"F1: {performance['f1']:.4f}")
print(f"AUC: {performance['auc']:.4f}")

# Optimization statistics
stats = results['optimization_stats']
print(f"Trials completed: {stats['trials_completed']}")
print(f"Total time: {stats['optimization_time']:.2f}s")
```

### Creating Tuned Model
```python
# Get best configuration after optimization
best_config = optimizer.get_best_config()

# Create model with best hyperparameters (if you provided model_factory)
tuned_model = optimizer.create_tuned_model(my_model_factory)

# Or use with custom model creation
def create_final_model(**params):
    # Your model creation logic using best params
    return model

final_model = create_final_model(**best_config)
```

## Built-in Capabilities

EasyHPO automatically handles many common tasks:

### Automatic Dataset Analysis
- **Time Series Detection**: Automatically detects if data is time series
- **Feature Analysis**: Analyzes univariate vs multivariate data
- **Class Imbalance**: Detects and handles imbalanced datasets
- **Complexity Assessment**: Evaluates dataset complexity for optimization strategy

### Smart Defaults
- **Model Architecture**: Provides sensible default models for common tasks
- **Training Logic**: Built-in training loops for standard cases
- **Evaluation Metrics**: Automatic F1, AUC, G-mean calculation
- **GPU Support**: Automatic CUDA detection and usage

### Robust Optimization
- **Multi-Agent RL**: Uses SQDDPG-based multi-agent optimization
- **Early Stopping**: Intelligent early stopping to prevent overfitting
- **Reproducibility**: Automatic seed management for consistent results
- **Error Handling**: Graceful degradation when components fail

## Complete Example: ECG Classification

```python
from MAT_HPO_LIB import EasyHPO
import numpy as np

# Simulate ECG data (in practice, load your real data)
n_samples, sequence_length = 1000, 500
X = np.random.randn(n_samples, sequence_length, 1)  # ECG signals
y = np.random.randint(0, 9, n_samples)              # 9 cardiac conditions

# Split data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                  stratify=y, random_state=42)

# Optimize with EasyHPO
optimizer = EasyHPO(
    task_type="ecg_classification",
    llm_enabled=True,              # Use LLM for intelligent suggestions
    max_trials=50,                 # More trials for better results
    verbose=True
)

# Run optimization
print("🚀 Starting ECG hyperparameter optimization...")
results = optimizer.optimize(X_train, y_train, X_val, y_val)

# Display results
print(f"\\n🏆 Best Performance:")
print(f"   F1 Score: {results['best_performance']['f1']:.4f}")
print(f"   AUC: {results['best_performance']['auc']:.4f}")
print(f"   G-mean: {results['best_performance']['gmean']:.4f}")

print(f"\\n⚙️ Best Hyperparameters:")
for param, value in results['best_hyperparameters'].items():
    print(f"   {param}: {value}")

print(f"\\n📊 Optimization completed in {results['optimization_time']:.2f} seconds")
```

## Tips for Best Results

1. **Data Quality**: Ensure your validation set is representative
2. **Trial Count**: Use 30-50 trials for good results, 100+ for production
3. **LLM Strategy**: Start with "adaptive" for most cases
4. **Task Type**: Choose the most specific task type available
5. **Custom Space**: Define custom hyperparameter space for specialized needs
6. **Timeout**: Set reasonable timeout for long-running optimizations

## Migration from Lower-Level APIs

If you're currently using the lower-level MAT_HPO_LIB APIs:

```python
# Old way (still supported)
from MAT_HPO_LIB import MAT_HPO_Optimizer, BaseEnvironment, HyperparameterSpace

class MyEnvironment(BaseEnvironment):
    # ... lots of boilerplate code

optimizer = MAT_HPO_Optimizer(environment, space, config)
results = optimizer.optimize()

# New way with EasyHPO
from MAT_HPO_LIB import EasyHPO

optimizer = EasyHPO(task_type="time_series_classification")
results = optimizer.optimize(X_train, y_train, X_val, y_val)
```

The EasyHPO interface reduces boilerplate while maintaining full power and flexibility when needed.