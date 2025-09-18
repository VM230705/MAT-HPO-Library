"""
LLM Configuration utilities for MAT_HPO_LIB
Provides easy-to-use configuration classes for LLM-enhanced optimization
"""

from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from .config import OptimizationConfig


@dataclass
class LLMClientConfig:
    """Configuration for LLM clients"""

    # Client type and connection
    client_type: str = "ollama"  # ollama, openai, anthropic
    model_name: str = "llama3.2:3b"
    base_url: Optional[str] = "http://localhost:11434"  # For Ollama
    api_key: Optional[str] = None  # For OpenAI/Anthropic

    # Request parameters
    temperature: float = 0.7
    max_tokens: Optional[int] = 1000
    timeout: int = 30

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class DatasetConfig:
    """Configuration for dataset analysis"""

    # Dataset identification
    dataset_name: str = "Unknown"
    dataset_info_csv_path: Optional[str] = None

    # Automatic analysis settings
    enable_auto_analysis: bool = True
    analysis_sample_size: int = 1000  # Max samples for analysis

    # Manual dataset characteristics (optional override)
    manual_characteristics: Optional[Dict[str, Any]] = None


@dataclass
class LLMOptimizationConfig:
    """Configuration for LLM-enhanced optimization"""

    # LLM integration settings
    enable_llm: bool = True
    llm_client_config: LLMClientConfig = field(default_factory=LLMClientConfig)

    # LLM activation strategy
    llm_strategy: str = "adaptive"  # fixed_alpha, adaptive, performance_based
    mixing_alpha: float = 0.3  # Weight for LLM suggestions (0=RL only, 1=LLM only)

    # Adaptive triggering parameters (when llm_strategy="adaptive")
    slope_threshold: float = 0.01  # Performance improvement threshold
    min_episodes_before_llm: int = 5  # Wait before first LLM call
    llm_cooldown_episodes: int = 5  # Episodes between LLM calls

    # Performance-based triggering (when llm_strategy="performance_based")
    performance_threshold: float = 0.8  # Trigger LLM if below this
    stagnation_episodes: int = 10  # Trigger LLM after stagnation

    # LLM prompt configuration
    custom_prompt_template: Optional[str] = None
    include_performance_history: bool = True
    include_dataset_analysis: bool = True
    max_history_episodes: int = 10


@dataclass
class TimeSeriesLLMConfig:
    """Complete configuration for time series LLM-enhanced HPO"""

    # Core optimization settings
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # LLM enhancement settings
    llm: LLMOptimizationConfig = field(default_factory=LLMOptimizationConfig)

    # Dataset settings
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Time series specific settings
    sequence_length: Optional[int] = None
    is_multivariate: Optional[bool] = None
    n_features: Optional[int] = None

    # Validation and early stopping
    validation_split: float = 0.1
    enable_early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_improvement: float = 0.001

    @classmethod
    def for_ecg_classification(cls,
                             dataset_name: str = "ECG",
                             n_classes: int = 9,
                             sequence_length: int = 1000,
                             enable_llm: bool = True) -> 'TimeSeriesLLMConfig':
        """Create configuration optimized for ECG classification"""

        config = cls()

        # Dataset configuration
        config.dataset.dataset_name = dataset_name
        config.sequence_length = sequence_length
        config.is_multivariate = False
        config.n_features = 1

        # Optimization configuration for ECG
        config.optimization.n_steps = 30
        config.optimization.n_agents = 3
        config.optimization.buffer_size = 1000

        # LLM configuration
        config.llm.enable_llm = enable_llm
        config.llm.llm_strategy = "adaptive"
        config.llm.mixing_alpha = 0.3

        # ECG-specific early stopping
        config.early_stopping_patience = 20
        config.early_stopping_min_improvement = 0.01

        return config

    @classmethod
    def for_generic_time_series(cls,
                               dataset_name: str = "TimeSeries",
                               n_classes: int = 2,
                               sequence_length: int = 100,
                               n_features: int = 1,
                               enable_llm: bool = True) -> 'TimeSeriesLLMConfig':
        """Create configuration for generic time series classification"""

        config = cls()

        # Dataset configuration
        config.dataset.dataset_name = dataset_name
        config.sequence_length = sequence_length
        config.is_multivariate = n_features > 1
        config.n_features = n_features

        # Optimization configuration
        config.optimization.n_steps = 20
        config.optimization.n_agents = 3
        config.optimization.buffer_size = 500

        # LLM configuration
        config.llm.enable_llm = enable_llm
        config.llm.llm_strategy = "adaptive"
        config.llm.mixing_alpha = 0.3

        return config

    @classmethod
    def quick_test_config(cls,
                         dataset_name: str = "QuickTest",
                         enable_llm: bool = True) -> 'TimeSeriesLLMConfig':
        """Create configuration for quick testing"""

        config = cls()

        # Dataset configuration
        config.dataset.dataset_name = dataset_name

        # Quick optimization configuration
        config.optimization.n_steps = 5
        config.optimization.n_agents = 3
        config.optimization.buffer_size = 100

        # LLM configuration
        config.llm.enable_llm = enable_llm
        config.llm.min_episodes_before_llm = 2
        config.llm.llm_cooldown_episodes = 2

        # Quick early stopping
        config.early_stopping_patience = 5

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        import dataclasses
        return {
            "optimization": dataclasses.asdict(self.optimization),
            "llm": dataclasses.asdict(self.llm),
            "dataset": dataclasses.asdict(self.dataset),
            "sequence_length": self.sequence_length,
            "is_multivariate": self.is_multivariate,
            "n_features": self.n_features,
            "validation_split": self.validation_split,
            "enable_early_stopping": self.enable_early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_improvement": self.early_stopping_min_improvement
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TimeSeriesLLMConfig':
        """Create configuration from dictionary"""
        config = cls()

        if "optimization" in config_dict:
            config.optimization = OptimizationConfig(**config_dict["optimization"])

        if "llm" in config_dict:
            llm_dict = config_dict["llm"]
            if "llm_client_config" in llm_dict:
                llm_dict["llm_client_config"] = LLMClientConfig(**llm_dict["llm_client_config"])
            config.llm = LLMOptimizationConfig(**llm_dict)

        if "dataset" in config_dict:
            config.dataset = DatasetConfig(**config_dict["dataset"])

        # Set remaining fields
        for field_name in ["sequence_length", "is_multivariate", "n_features",
                          "validation_split", "enable_early_stopping",
                          "early_stopping_patience", "early_stopping_min_improvement"]:
            if field_name in config_dict:
                setattr(config, field_name, config_dict[field_name])

        return config


def create_llm_config(dataset_name: str,
                     task_type: str = "ecg_classification",
                     enable_llm: bool = True,
                     llm_model: str = "llama3.2:3b",
                     n_steps: int = 30,
                     **kwargs) -> TimeSeriesLLMConfig:
    """
    Quick configuration creation utility

    Args:
        dataset_name: Name of the dataset
        task_type: Type of task ("ecg_classification", "generic_time_series", "quick_test")
        enable_llm: Whether to enable LLM
        llm_model: LLM model to use
        n_steps: Number of optimization steps
        **kwargs: Additional configuration parameters

    Returns:
        TimeSeriesLLMConfig instance
    """

    if task_type == "ecg_classification":
        config = TimeSeriesLLMConfig.for_ecg_classification(
            dataset_name=dataset_name,
            enable_llm=enable_llm
        )
    elif task_type == "generic_time_series":
        config = TimeSeriesLLMConfig.for_generic_time_series(
            dataset_name=dataset_name,
            enable_llm=enable_llm
        )
    elif task_type == "quick_test":
        config = TimeSeriesLLMConfig.quick_test_config(
            dataset_name=dataset_name,
            enable_llm=enable_llm
        )
    else:
        config = TimeSeriesLLMConfig()
        config.dataset.dataset_name = dataset_name
        config.llm.enable_llm = enable_llm

    # Apply custom settings
    config.optimization.n_steps = n_steps
    config.llm.llm_client_config.model_name = llm_model

    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.optimization, key):
            setattr(config.optimization, key, value)
        elif hasattr(config.llm, key):
            setattr(config.llm, key, value)
        elif hasattr(config.dataset, key):
            setattr(config.dataset, key, value)

    return config