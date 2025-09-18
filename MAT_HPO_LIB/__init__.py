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
"""

# Core components
from .core.multi_agent_optimizer import MAT_HPO_Optimizer
from .core.multi_fidelity_optimizer import MultiFidelityMAT_HPO, FidelityConfig, MultiFidelityEnvironment
from .core.llm_enhanced_optimizer import LLMEnhancedMAT_HPO_Optimizer, LLMEnhancedOptimizationConfig, LLMConfigs
from .core.base_environment import BaseEnvironment
from .core.enhanced_environment import TimeSeriesEnvironment
from .core.hyperparameter_space import HyperparameterSpace

# Easy-to-use high-level interface
from .easy_hpo import EasyHPO
from .full_control_hpo import FullControlHPO
from .configurable_optimizer import LLMEnhancedHPO

# LLM components
from .llm import (
    BaseLLMClient,
    DefaultJSONParser,
    OllamaLLMClient,
    LLMHyperparameterMixer,
    EnhancedLLMHyperparameterMixer,
    AdaptiveAlphaController,
    LLaPipeAdaptiveAdvisor,
    PerformanceMetricCalculator,
    DatasetInfoReader,
    get_enhanced_dataset_info,
    get_dataset_recommendations,
    analyze_time_series_dataset,
    get_time_series_llm_context,
    get_universal_dataset_info,
    TimeSeriesAnalyzer,
    LLMConversationLogger
)

# Try to import optional LLM clients
try:
    from .llm import OpenAILLMClient, AzureOpenAILLMClient
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    from .llm import AnthropicLLMClient, ClaudeLegacyClient
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

# Utilities
from .utils.config import OptimizationConfig, DefaultConfigs
from .utils.llm_config import TimeSeriesLLMConfig, LLMClientConfig, DatasetConfig, LLMOptimizationConfig, create_llm_config
from .utils.metrics import MetricsCalculator, calculate_f1_auc_gmean
from .utils.logger import HPOLogger, SimpleLogger

__version__ = "1.1.0"
__author__ = "MAT-HPO Development Team"

__all__ = [
    # Core components
    "MAT_HPO_Optimizer",
    "MultiFidelityMAT_HPO",
    "FidelityConfig",
    "MultiFidelityEnvironment",
    "LLMEnhancedMAT_HPO_Optimizer",
    "LLMEnhancedOptimizationConfig",
    "LLMConfigs",
    "BaseEnvironment",
    "TimeSeriesEnvironment",
    "HyperparameterSpace",
    # Easy-to-use interface
    "EasyHPO",
    "FullControlHPO",
    "LLMEnhancedHPO",
    # LLM components - Core
    "BaseLLMClient",
    "DefaultJSONParser",
    "OllamaLLMClient",
    "LLMHyperparameterMixer",
    "EnhancedLLMHyperparameterMixer",
    "AdaptiveAlphaController",
    "LLaPipeAdaptiveAdvisor",
    "PerformanceMetricCalculator",
    "DatasetInfoReader",
    "get_enhanced_dataset_info",
    "get_dataset_recommendations",
    "analyze_time_series_dataset",
    "get_time_series_llm_context",
    "get_universal_dataset_info",
    "TimeSeriesAnalyzer",
    "LLMConversationLogger",
    # Configuration
    "OptimizationConfig",
    "DefaultConfigs",
    "TimeSeriesLLMConfig",
    "LLMClientConfig",
    "DatasetConfig",
    "LLMOptimizationConfig",
    "create_llm_config",
    # Utilities
    "MetricsCalculator",
    "calculate_f1_auc_gmean",
    "HPOLogger",
    "SimpleLogger"
]

# Add optional LLM clients to __all__ if available
if _OPENAI_AVAILABLE:
    __all__.extend(["OpenAILLMClient", "AzureOpenAILLMClient"])

if _ANTHROPIC_AVAILABLE:
    __all__.extend(["AnthropicLLMClient", "ClaudeLegacyClient"])