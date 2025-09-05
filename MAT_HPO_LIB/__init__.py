"""
MAT-HPO Library: Multi-Agent Transformer Hyperparameter Optimization

A sophisticated library for hyperparameter optimization using multi-agent reinforcement learning.
Built on the SQDDPG algorithm with Shapley value-based credit assignment.

Main Components:
- MAT_HPO_Optimizer: Main optimization interface
- BaseEnvironment: Abstract base for optimization environments
- HyperparameterSpace: Define search space for parameters
- OptimizationConfig: Configuration management with presets
- MetricsCalculator: Performance evaluation utilities

Quick Start:
    from MAT_HPO_LIB import MAT_HPO_Optimizer, BaseEnvironment, HyperparameterSpace
    from MAT_HPO_LIB.utils import DefaultConfigs
    
    # Define your environment
    class MyEnvironment(BaseEnvironment):
        def step(self, hyperparams):
            # Your optimization logic here
            return f1, auc, gmean, done
    
    # Set up optimization
    space = HyperparameterSpace()
    space.add_continuous('learning_rate', 1e-4, 1e-2, agent=0)
    space.add_discrete('batch_size', [16, 32, 64], agent=1)
    
    config = DefaultConfigs.standard()
    optimizer = MAT_HPO_Optimizer(environment, space, config)
    results = optimizer.optimize()
"""

# Core components
from .core.multi_agent_optimizer import MAT_HPO_Optimizer
from .core.multi_fidelity_optimizer import MultiFidelityMAT_HPO, FidelityConfig, MultiFidelityEnvironment
from .core.base_environment import BaseEnvironment
from .core.hyperparameter_space import HyperparameterSpace

# Utilities
from .utils.config import OptimizationConfig, DefaultConfigs
from .utils.metrics import MetricsCalculator, calculate_f1_auc_gmean
from .utils.logger import HPOLogger, SimpleLogger

__version__ = "1.0.0"
__author__ = "MAT-HPO Development Team"

__all__ = [
    # Core components
    "MAT_HPO_Optimizer",
    "MultiFidelityMAT_HPO",
    "FidelityConfig", 
    "MultiFidelityEnvironment",
    "BaseEnvironment", 
    "HyperparameterSpace",
    # Configuration
    "OptimizationConfig",
    "DefaultConfigs",
    # Utilities
    "MetricsCalculator",
    "calculate_f1_auc_gmean",
    "HPOLogger",
    "SimpleLogger"
]