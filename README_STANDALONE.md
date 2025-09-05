# MAT-HPO Library

A Multi-Agent Transformer Hyperparameter Optimization library for machine learning models.

## Overview

MAT-HPO is a production-ready library that implements multi-agent reinforcement learning for hyperparameter optimization. It uses three specialized agents working collaboratively to optimize different aspects of machine learning models:

- **Agent 0**: Problem-specific parameters (class weights, regularization)
- **Agent 1**: Model architecture parameters (hidden size, layers)
- **Agent 2**: Training parameters (batch size, learning rate)

## Key Features

- ✅ **Multi-Agent Architecture**: 3 specialized RL agents using SQDDPG
- ✅ **Flexible Integration**: Abstract base environment for easy customization
- ✅ **Production Ready**: Comprehensive error handling and logging
- ✅ **Memory Efficient**: Optimized for large-scale hyperparameter spaces
- ✅ **Well Documented**: Complete usage guides and examples

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd MAT-HPO-Library

# Install dependencies
pip install torch torchvision numpy scikit-learn tqdm wandb
```

## Quick Start

```python
from MAT_HPO_LIB import MAT_HPO_Optimizer, BaseEnvironment, HyperparameterSpace

class YourEnvironment(BaseEnvironment):
    def load_data(self):
        # Your data loading logic
        return train_data, val_data, test_data
    
    def create_model(self, hyperparams):
        # Your model creation logic
        return model
    
    def train_evaluate(self, model, hyperparams):
        # Your training and evaluation logic
        return {'f1': f1_score, 'accuracy': accuracy}
    
    def compute_reward(self, metrics):
        # Your reward computation logic
        return metrics['f1']

# Create hyperparameter space
space = HyperparameterSpace()
space.add_hyperparameter('learning_rate', 'numerical', [1e-5, 1e-3], 1e-4)
space.add_hyperparameter('batch_size', 'numerical', [16, 128], 32)

# Initialize and run optimization
environment = YourEnvironment()
optimizer = MAT_HPO_Optimizer(environment, space, config)
results = optimizer.optimize()
```

## Architecture

### Core Components

- **`core/base_environment.py`**: Abstract base class for environments
- **`core/multi_agent_optimizer.py`**: Main optimization engine
- **`core/hyperparameter_space.py`**: Hyperparameter space definition
- **`utils/`**: Utility functions and helpers
- **`examples/`**: Usage examples and tutorials

### Agent Specialization

Each agent is specialized for different parameter types:

```python
# Agent 0: Problem-specific parameters
space.assign_to_agent(0, 'class_weight_0')
space.assign_to_agent(0, 'regularization')

# Agent 1: Architecture parameters  
space.assign_to_agent(1, 'hidden_size')
space.assign_to_agent(1, 'num_layers')

# Agent 2: Training parameters
space.assign_to_agent(2, 'batch_size')
space.assign_to_agent(2, 'learning_rate')
```

## Documentation

- **[Detailed Usage Guide](DETAILED_USAGE_GUIDE.md)**: Comprehensive tutorial
- **[API Reference](docs/api_reference.md)**: Complete API documentation
- **[Examples](examples/)**: Working examples for different use cases

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_library.py -v
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{mat_hpo_library,
  title = {MAT-HPO: Multi-Agent Transformer Hyperparameter Optimization Library},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-username/MAT-HPO-Library}
}
```

## Acknowledgments

- Built for the SPNV2 ECG classification project
- Implements SQDDPG (Soft Q-learning with Double Deep Policy Gradients)
- Designed for production machine learning workflows