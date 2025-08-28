# MAT-HPO Library

**Multi-Agent Transformer Hyperparameter Optimization Library**

A flexible and powerful library for hyperparameter optimization using multi-agent reinforcement learning. Built on the MAT-HPO framework, this library can optimize hyperparameters for any machine learning model or pipeline.

## 🚀 Key Features

- **Multi-Agent Architecture**: Three specialized agents optimize different types of hyperparameters simultaneously
- **Transformer-Based**: Uses transformer neural networks for intelligent hyperparameter search
- **Flexible Interface**: Easy to integrate with any ML model or training pipeline
- **CUDA Support**: GPU acceleration for faster optimization
- **Comprehensive Logging**: Detailed logging and result tracking
- **Extensible Design**: Abstract base classes allow easy customization

## 🏗️ Architecture

The library uses a multi-agent approach where three agents specialize in different hyperparameter types:

- **Agent 0**: Problem-specific parameters (e.g., class weights, regularization)
- **Agent 1**: Model architecture parameters (e.g., network dimensions, layers)
- **Agent 2**: Training parameters (e.g., batch size, learning rate)

Each agent uses transformer networks to learn optimal hyperparameter selection strategies through reinforcement learning.

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd MAT_HPO_LIB

# Install dependencies
pip install torch numpy scikit-learn
```

## 🔧 Quick Start

### Basic Usage

```python
from MAT_HPO_LIB import MAT_HPO_Optimizer, BaseEnvironment, HyperparameterSpace, OptimizationConfig

# 1. Define your environment
class MyEnvironment(BaseEnvironment):
    def load_data(self):
        # Load your dataset
        return data
    
    def create_model(self, hyperparams):
        # Create model with hyperparameters
        return model
    
    def train_evaluate(self, model, hyperparams):
        # Train and evaluate model
        return {'f1': f1_score, 'accuracy': accuracy}
    
    def compute_reward(self, metrics):
        # Compute reward from metrics
        return reward

# 2. Define hyperparameter space
hyperparameter_space = HyperparameterSpace(
    agent0_params=['param1', 'param2'],  # Agent 0 parameters
    agent1_params=['param3', 'param4'],  # Agent 1 parameters  
    agent2_params=['param5', 'param6'],  # Agent 2 parameters
    bounds={'param1': (0.1, 10.0), ...},  # Parameter bounds
    param_types={'param1': float, ...}     # Parameter types
)

# 3. Create and run optimizer
optimizer = MAT_HPO_Optimizer(
    environment=MyEnvironment(),
    hyperparameter_space=hyperparameter_space,
    config=OptimizationConfig(max_steps=100)
)

results = optimizer.optimize()
best_params = results['best_hyperparameters']
```

### SPL Example

For optimizing the SPL (SPNV2) heart disease classification system:

```python
from MAT_HPO_LIB.examples.spl_hpo_example import SPLEnvironment, create_spl_hyperparameter_space

# Create SPL environment
environment = SPLEnvironment(
    dataset='ptbxl',
    fold=1,
    gpu='0',
    num_classes=5
)

# Create hyperparameter space
hyperparameter_space = create_spl_hyperparameter_space(num_classes=5)

# Run optimization
optimizer = MAT_HPO_Optimizer(environment, hyperparameter_space, config)
results = optimizer.optimize()
```

## 📝 Examples

### 1. Simple Synthetic Example

```bash
cd examples
python simple_example.py
```

Demonstrates the library with a synthetic optimization problem. Good for understanding basic functionality.

### 2. SPL Integration Example

```bash
cd examples  
python spl_hpo_example.py
```

Shows how to integrate MAT-HPO with the SPL (SPNV2) heart disease classification system.

## 🔍 Core Components

### BaseEnvironment

Abstract base class that you inherit to define your optimization problem:

```python
class BaseEnvironment(ABC):
    @abstractmethod
    def load_data(self): pass
    
    @abstractmethod
    def create_model(self, hyperparams): pass
    
    @abstractmethod
    def train_evaluate(self, model, hyperparams): pass
    
    @abstractmethod
    def compute_reward(self, metrics): pass
```

### HyperparameterSpace

Defines the search space for hyperparameters:

```python
hyperparameter_space = HyperparameterSpace(
    agent0_params=['class_weight_0', 'class_weight_1'],
    agent1_params=['hidden_size', 'num_layers'], 
    agent2_params=['batch_size', 'learning_rate'],
    bounds={
        'class_weight_0': (0.1, 10.0),
        'hidden_size': (64, 512),
        'batch_size': (16, 128),
        'learning_rate': (1e-5, 1e-2)
    },
    param_types={
        'class_weight_0': float,
        'hidden_size': int,
        'batch_size': int,
        'learning_rate': float
    }
)
```

### OptimizationConfig

Configuration for the optimization process:

```python
config = OptimizationConfig(
    max_steps=100,
    replay_buffer_size=1000,
    batch_size=32,
    policy_learning_rate=1e-4,
    value_learning_rate=1e-3,
    use_cuda=True,
    gpu_device=0
)
```

## 📊 Results and Logging

The optimizer automatically saves:

- **Best hyperparameters**: `best_hyperparams.json`
- **Optimization log**: `optimization_log.txt`
- **Step-by-step data**: `step_log.jsonl`
- **Final results**: `optimization_results.json`
- **Model checkpoints**: `best_actor*.pt`

## ⚙️ Configuration Options

### Quick Configurations

```python
from MAT_HPO_LIB.utils.config import DefaultConfigs

# Quick test (10 steps)
config = DefaultConfigs.quick_test()

# Standard optimization (100 steps)
config = DefaultConfigs.standard()

# Extensive optimization (500 steps)
config = DefaultConfigs.extensive()

# CPU-only optimization
config = DefaultConfigs.cpu_only()
```

### Custom Configuration

```python
config = OptimizationConfig(
    max_steps=200,
    replay_buffer_size=2000,
    batch_size=64,
    early_stop_patience=30,
    gradient_clip=1.0,
    save_interval=20,
    verbose=True
)
```

## 🔬 Advanced Usage

### Custom Reward Functions

```python
def compute_reward(self, metrics):
    # Custom weighted reward
    f1_weight = 0.5
    acc_weight = 0.3
    efficiency_weight = 0.2
    
    return (metrics['f1'] * f1_weight + 
            metrics['accuracy'] * acc_weight +
            (1.0 - metrics['training_time'] / max_time) * efficiency_weight)
```

### Early Stopping

```python
config = OptimizationConfig(
    early_stop_patience=20,      # Stop if no improvement for 20 steps
    early_stop_threshold=1e-4    # Minimum improvement threshold
)
```

### Custom Metrics

```python
def train_evaluate(self, model, hyperparams):
    # Train your model
    trained_model = self.train_model(model, hyperparams)
    
    # Evaluate with custom metrics
    predictions = trained_model.predict(self.test_data)
    
    return {
        'accuracy': accuracy_score(self.test_labels, predictions),
        'f1': f1_score(self.test_labels, predictions, average='weighted'),
        'custom_metric': my_custom_metric(self.test_labels, predictions),
        'training_time': training_time
    }
```

## 🚨 Error Handling

The library includes robust error handling:

- **Timeout protection**: Prevents hanging during model training
- **Graceful degradation**: Returns poor scores if training fails
- **Automatic cleanup**: Cleans up temporary files
- **Detailed logging**: Logs errors for debugging

## 🔧 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```python
   config = OptimizationConfig(use_cuda=False)  # Use CPU
   # or reduce batch size
   config.batch_size = 16
   ```

2. **Import errors**
   ```bash
   # Make sure you're in the correct directory
   export PYTHONPATH="${PYTHONPATH}:/path/to/MAT_HPO_LIB"
   ```

3. **Training script not found**
   ```python
   # Use absolute path
   environment = SPLEnvironment(spl_script_path='/absolute/path/to/SPL.py')
   ```

## 📈 Performance Tips

1. **Use GPU acceleration** when available
2. **Adjust replay buffer size** based on available memory
3. **Start with quick_test config** for initial validation
4. **Use early stopping** to avoid overoptimization
5. **Monitor step times** and adjust timeout values

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this library in your research, please cite:

```bibtex
@article{mat_hpo_lib,
    title={MAT-HPO: Multi-Agent Transformer Hyperparameter Optimization Library},
    author={[Your Name]},
    year={2024},
    journal={[Journal Name]}
}
```

## 🔗 Related Work

- Original MAT-HPO paper: [Link to paper]
- SQDDPG algorithm: [Link to algorithm]
- Applications in time series classification: [Link to applications]

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the examples directory
- Review the documentation

---

**MAT-HPO Library** - Making hyperparameter optimization smarter, faster, and more accessible! 🚀