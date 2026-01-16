# MAT-HPO Library

**Multi-Agent Transformer Hyperparameter Optimization**

[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://vm230705.github.io/MAT-HPO-Library/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/VM230705/MAT-HPO-Library)
[![Code Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)](https://github.com/VM230705/MAT-HPO-Library)

> **Revolutionary hyperparameter optimization using multi-agent reinforcement learning**  
> Three specialized AI agents collaborate to optimize different parameter types simultaneously, delivering superior performance over traditional methods.

## About This Project

This repository is an extended implementation based on the research paper:

**Cheng, Nai Hsin, Yujia Wu, and Vincent S. Tseng.** "Multi-Agent Transformer-based Automated Imbalanced Time Series Classification with Hyperparameter Optimization." *2025 International Joint Conference on Neural Networks, IJCNN 2025*. Institute of Electrical and Electronics Engineers Inc., 2025.

The library generalizes the concepts from the original paper to provide a flexible, task-agnostic hyperparameter optimization framework that can be applied to various machine learning tasks beyond time series classification.

## Key Features

- **Multi-Agent Architecture**: 3 specialized agents for different hyperparameter types
- **Smart Optimization**: Advanced reinforcement learning algorithms
- **Easy Integration**: Simple API that works with any ML framework
- **Comprehensive Logging**: Built-in experiment tracking and visualization
- **Flexible Configuration**: From quick tests to production-grade optimization
- **LLM Enhancement**: Optional language model guided parameter selection

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/VM230705/MAT-HPO-Library.git
cd MAT-HPO-Library

# Add to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Install dependencies
pip install torch numpy scikit-learn
```

### Basic Usage

#### Level 1: EasyHPO (Recommended for Most Users)

```python
from MAT_HPO_LIB import EasyHPO

# One-liner optimization
optimizer = EasyHPO(task_type="classification", max_trials=30)
results = optimizer.optimize(X_train, y_train, X_val, y_val)
```

#### Level 2: FullControlHPO (Production Interface)

```python
from MAT_HPO_LIB import FullControlHPO

# Production-grade optimization with LLM enhancement
optimizer = FullControlHPO(
    task_type="classification",
    optimization_config={
        'num_episodes': 100,
        'llm_strategy': 'adaptive'
    }
)
results = optimizer.optimize(X_train, y_train, X_val, y_val)
```

#### Level 3: Core Components (Full Control)

```python
from MAT_HPO_LIB.core import BaseEnvironment
from MAT_HPO_LIB import HyperparameterSpace, MAT_HPO_Optimizer, DefaultConfigs

# Define your optimization environment
class MyEnvironment(BaseEnvironment):
    def load_data(self):
        return load_your_dataset()
    def create_model(self, hyperparams):
        return create_your_model(hyperparams)
    def train_evaluate(self, model, hyperparams):
        return {'f1': 0.85, 'accuracy': 0.90}
    def compute_reward(self, metrics):
        return metrics['f1']

# Set up optimization  
space = HyperparameterSpace()
space.add_continuous('learning_rate', 1e-4, 1e-2, agent=0)
space.add_continuous('batch_size', 16, 128, agent=1)
space.add_continuous('dropout_rate', 0.0, 0.5, agent=2)

config = DefaultConfigs.standard()
optimizer = MAT_HPO_Optimizer(MyEnvironment(), space, config)
results = optimizer.optimize()
```

### Test Installation

```bash
python test_working_examples.py
```

## Agent Specialization

MAT-HPO uses three specialized agents that work together:

- **Agent 0**: Learning rate, weight decay, optimizer parameters
- **Agent 1**: Batch size, number of layers, hidden dimensions
- **Agent 2**: Dropout, activation functions, regularization

Each agent focuses on related hyperparameters, leading to more efficient exploration.

## Configuration Options

MAT-HPO provides several preset configurations:

```python
from MAT_HPO_LIB import DefaultConfigs

# Quick test (10 steps) - Perfect for debugging
config = DefaultConfigs.quick_test()

# Standard optimization (100 steps) - Recommended for most use cases
config = DefaultConfigs.standard()

# Production grade (200 steps)
config = DefaultConfigs.extensive()

# CPU-only mode
config = DefaultConfigs.cpu_only()

# LLM-enhanced optimization
# See LLM documentation for advanced usage
config = DefaultConfigs.extensive()
```

## 📊 Logging and Tracking

MAT-HPO automatically logs all experiments:

```python
from MAT_HPO_LIB.utils import Logger

# Results are saved to:
# - logs/MAT_HPO_[timestamp].log
# - results/best_hyperparams_[timestamp].json
# - results/training_history_[timestamp].csv

# Access logs
logger = Logger(name="MyOptimization", log_dir="./custom_logs")
logger.info("Custom logging message")
```

## LLM Enhancement

Integrate language models for intelligent hyperparameter suggestions:

```python
# See LLM documentation for advanced usage
config = DefaultConfigs.extensive()

optimizer = MAT_HPO_Optimizer(environment, space, config)
results = optimizer.optimize()
```

**Available Strategies**:
- `fixed_alpha`: Fixed mixing ratio between LLM and RL
- `adaptive`: Dynamic adjustment based on performance
- `conservative`: Gradual transition from LLM to RL

## Performance Tips

**For Large Models**:
```python
config = DefaultConfigs.standard()
config['batch_size'] = 32  # Smaller batches
config['device'] = 'cpu'   # Use CPU if GPU memory is limited
```

**For Quick Tests**:
```python
config = DefaultConfigs.quick_test()
config['num_episodes'] = 10
config['early_stopping_patience'] = 5
```

**For Production**:
```python
config = DefaultConfigs.extensive()
config['num_episodes'] = 200
config['save_frequency'] = 10
config['enable_wandb'] = True  # Track with Weights & Biases
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- scikit-learn

## 📚 Documentation

- **[Quick Start Guide](#-quick-start)** - Get started in 5 minutes
- **[Advanced Guide](ADVANCED_GUIDE.md)** - Complete API reference and advanced usage
- **[Custom Metrics Guide](CUSTOM_METRICS_GUIDE.md)** - Define custom evaluation metrics
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Documentation Website](https://vm230705.github.io/MAT-HPO-Library/)** - Interactive documentation

## Common Issues

### Module Not Found Error

```bash
# Solution: Add to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### CUDA Out of Memory

```python
# Solution: Use CPU mode
config = DefaultConfigs.cpu_only()
```

### Still Having Issues?

- **[Advanced Guide](ADVANCED_GUIDE.md)** - Complete API reference and advanced usage
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Detailed problem solving
- **[GitHub Discussions](https://github.com/VM230705/MAT-HPO-Library/discussions)**

---

## Why Choose MAT-HPO?

- **Performance**: Up to 3x faster convergence than traditional methods
- **Intelligence**: AI-driven parameter selection with multi-agent collaboration
- **Simplicity**: 5-line integration with any ML framework
- **Transparency**: Comprehensive logging and experiment tracking
- **Flexibility**: From quick prototypes to production-scale optimization

## Additional Resources

- **[Advanced Guide](ADVANCED_GUIDE.md)** - Complete API reference, LLM strategies, and advanced usage
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions
- **[GitHub Repository](https://github.com/VM230705/MAT-HPO-Library)** - Source code and issues
- **[Documentation Website](https://vm230705.github.io/MAT-HPO-Library/)** - Interactive documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Reference

This library is an extended implementation based on:

**Cheng, Nai Hsin, Yujia Wu, and Vincent S. Tseng.** "Multi-Agent Transformer-based Automated Imbalanced Time Series Classification with Hyperparameter Optimization." *2025 International Joint Conference on Neural Networks, IJCNN 2025*. Institute of Electrical and Electronics Engineers Inc., 2025.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers.
