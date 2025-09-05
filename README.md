# MAT-HPO Library

Multi-Agent Transformer Hyperparameter Optimization

[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://vm230705.github.io/MAT-HPO-Library/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A library for hyperparameter optimization using multi-agent reinforcement learning. Three specialized agents optimize different parameter types simultaneously.

## Quick Start

### Installation
```bash
git clone https://github.com/VM230705/MAT-HPO-Library.git
cd MAT-HPO-Library
pip install torch numpy scikit-learn
```

### Basic Usage
```python
from MAT_HPO_LIB import MAT_HPO_Optimizer, BaseEnvironment, HyperparameterSpace
from MAT_HPO_LIB.utils import DefaultConfigs

# 1. Define your optimization environment
class MyEnvironment(BaseEnvironment):
    def __init__(self, name="MyEnvironment"):
        super().__init__(name)
        
    def step(self, hyperparams):
        # Your optimization logic here
        # Train model, evaluate, return metrics
        f1, auc, gmean = train_and_evaluate(hyperparams)
        done = check_stopping_condition()
        return f1, auc, gmean, done

# 2. Define hyperparameter space
space = HyperparameterSpace()
space.add_continuous('learning_rate', 1e-5, 1e-2, agent=0)
space.add_discrete('batch_size', [16, 32, 64, 128], agent=1)
space.add_continuous('dropout', 0.0, 0.5, agent=2)

# 3. Run optimization
config = DefaultConfigs.standard()  # 100 steps
optimizer = MAT_HPO_Optimizer(environment, space, config)
results = optimizer.optimize()

print("Best hyperparameters:", results['best_hyperparameters'])
```

### Test Installation
```bash
python simple_functionality_test.py
```

## Agent Specialization

- **Agent 0**: Problem-specific parameters (class weights, regularization)
- **Agent 1**: Architecture parameters (hidden layers, network size)  
- **Agent 2**: Training parameters (batch size, learning rate)

## Configuration Options

```python
from MAT_HPO_LIB.utils import DefaultConfigs

# Quick test (10 steps)
config = DefaultConfigs.quick_test()

# Standard optimization (100 steps) 
config = DefaultConfigs.standard()

# CPU-only mode
config = DefaultConfigs.cpu_only()

# Custom configuration
from MAT_HPO_LIB.utils import OptimizationConfig
config = OptimizationConfig(
    max_steps=50,
    batch_size=32,
    use_cuda=True
)
```

## API Reference

### HyperparameterSpace
```python
space = HyperparameterSpace()

# Continuous parameters
space.add_continuous(name, min_val, max_val, agent)

# Discrete parameters  
space.add_discrete(name, choices_list, agent)

# Boolean parameters
space.add_boolean(name, agent)
```

### BaseEnvironment
```python
class YourEnvironment(BaseEnvironment):
    def step(self, hyperparams):
        # Return: f1_score, auc_score, gmean_score, done
        return f1, auc, gmean, done
```

## Results

Results are automatically saved:
- `best_hyperparams.json` - Best hyperparameters found
- `optimization_results.json` - Complete optimization results
- `step_log.jsonl` - Step-by-step optimization log

## Troubleshooting

**Import Error:**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**CUDA Out of Memory:**
```python
config = DefaultConfigs.cpu_only()
```

**Test Library:**
```bash
python simple_functionality_test.py
```

📖 **[Full Documentation](https://vm230705.github.io/MAT-HPO-Library/)**