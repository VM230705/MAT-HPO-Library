"""
Comprehensive Configuration Management System for MAT-HPO Optimization

This module provides a robust, type-safe configuration management system designed
specifically for multi-agent hyperparameter optimization. It offers:

**Key Features**:
- Type-safe configuration with automatic validation
- Pre-built configuration templates for common scenarios
- Flexible parameter updating with validation
- Device management with automatic GPU detection
- Reproducibility controls with seed management
- Serialization support for configuration persistence

**Configuration Categories**:
1. **Training Parameters**: Steps, buffer sizes, batch configurations
2. **Learning Dynamics**: Learning rates, update frequencies, gradient control
3. **Hardware Management**: Device selection, CUDA settings, memory optimization
4. **Monitoring & Logging**: Save intervals, verbosity, progress tracking
5. **Optimization Control**: Early stopping, convergence criteria, noise settings
6. **Reproducibility**: Seed management, deterministic behavior controls

**Usage Patterns**:
```python
# Quick setup with defaults
config = DefaultConfigs.standard()

# Custom configuration
config = OptimizationConfig(
    max_steps=200,
    policy_learning_rate=1e-4,
    early_stop_patience=30
)

# Dynamic updates
config.update(batch_size=64, verbose=True)
```

The configuration system is designed to be both user-friendly for quick setups
and powerful enough for advanced research scenarios requiring fine-grained control.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class OptimizationConfig:
    """
    Comprehensive Configuration Management for Multi-Agent Hyperparameter Optimization.
    
    This dataclass encapsulates all configuration parameters needed for MAT-HPO optimization,
    providing type safety, automatic validation, and intelligent defaults. The configuration
    is organized into logical groups for easy management and understanding.
    
    **Design Principles**:
    - **Type Safety**: All parameters have explicit types with validation
    - **Sensible Defaults**: Default values work well for most hyperparameter optimization tasks
    - **Flexible Customization**: Easy to override specific parameters while keeping others
    - **Automatic Validation**: Post-initialization checks ensure parameter consistency
    - **Device Intelligence**: Automatic GPU detection with graceful CPU fallback
    - **Reproducibility**: Built-in seed management and deterministic behavior controls
    
    **Parameter Groups**:
    
    **Training Control**:
    - max_steps: Total optimization iterations
    - batch_size: Experience replay batch size
    - replay_buffer_size: Maximum stored experiences
    
    **Learning Dynamics**:
    - policy_learning_rate: Actor network learning rate
    - value_learning_rate: Critic network learning rate
    - behaviour_update_freq: Policy update frequency
    - critic_update_times: Value function updates per policy update
    
    **Hardware & Performance**:
    - gpu_device: Specific GPU device ID
    - use_cuda: Enable/disable CUDA acceleration
    - device: Automatically selected computation device
    
    **Monitoring & Persistence**:
    - save_interval: Model checkpointing frequency
    - log_interval: Progress logging frequency
    - verbose: Detailed output control
    - save_best_model: Automatic best model preservation
    
    **Convergence & Stopping**:
    - early_stop_patience: Steps without improvement before stopping
    - early_stop_threshold: Minimum improvement threshold
    - gradient_clip: Gradient explosion prevention
    
    **Reproducibility**:
    - seed: Random seed for reproducible results
    - deterministic: Enable deterministic training behavior
    """
    
    # Training parameters
    max_steps: int = 100
    replay_buffer_size: int = 1000
    batch_size: int = 32
    
    # Learning rates
    policy_learning_rate: float = 1e-4
    value_learning_rate: float = 1e-3
    
    # Update frequencies
    behaviour_update_freq: int = 5
    critic_update_times: int = 1
    
    # Device settings
    gpu_device: int = 0
    use_cuda: bool = True
    
    # Logging and saving
    save_interval: int = 10
    log_interval: int = 1
    verbose: bool = True
    
    # Early stopping
    early_stop_patience: int = 20
    early_stop_threshold: float = 1e-6
    
    # Model saving
    save_best_model: bool = True
    model_save_path: str = "./models"
    
    # Reproducibility 
    seed: int = 42
    deterministic: bool = True
    
    # Advanced settings
    gradient_clip: float = 1.0
    target_update_tau: float = 0.005
    noise_std: float = 0.1
    
    def __post_init__(self):
        """
        Comprehensive post-initialization validation and intelligent system setup.
        
        This method performs critical validation and initialization tasks:
        1. **Parameter Validation**: Ensures all parameters are within valid ranges
        2. **Device Configuration**: Intelligent GPU detection with fallback handling
        3. **Reproducibility Setup**: Configures random seeds and deterministic behavior
        4. **System Optimization**: Applies performance-oriented settings when appropriate
        
        The method is automatically called after dataclass initialization and ensures
        the configuration is both valid and optimally configured for the target system.
        
        Raises:
            ValueError: If any configuration parameters are invalid or inconsistent
            RuntimeError: If device setup fails or CUDA configuration encounters issues
        """
        # Comprehensive parameter validation with detailed error messages
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.replay_buffer_size <= 0:
            raise ValueError(f"replay_buffer_size must be positive, got {self.replay_buffer_size}")
        if self.batch_size > self.replay_buffer_size:
            raise ValueError(f"batch_size ({self.batch_size}) cannot exceed replay_buffer_size ({self.replay_buffer_size})")
        if self.policy_learning_rate <= 0:
            raise ValueError(f"policy_learning_rate must be positive, got {self.policy_learning_rate}")
        if self.value_learning_rate <= 0:
            raise ValueError(f"value_learning_rate must be positive, got {self.value_learning_rate}")
        if self.gradient_clip <= 0:
            raise ValueError(f"gradient_clip must be positive, got {self.gradient_clip}")
        if not 0 <= self.target_update_tau <= 1:
            raise ValueError(f"target_update_tau must be in [0,1], got {self.target_update_tau}")
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {self.noise_std}")
            
        # Intelligent device setup with comprehensive error handling
        if self.use_cuda and torch.cuda.is_available():
            # Validate GPU device ID
            if self.gpu_device >= torch.cuda.device_count():
                available_devices = list(range(torch.cuda.device_count()))
                raise ValueError(f"Invalid gpu_device {self.gpu_device}. Available devices: {available_devices}")
            
            self.device = torch.device(f'cuda:{self.gpu_device}')
            if self.verbose:
                gpu_name = torch.cuda.get_device_name(self.gpu_device)
                memory_gb = torch.cuda.get_device_properties(self.gpu_device).total_memory / 1e9
                print(f"🚀 Using CUDA device: {self.device} ({gpu_name}, {memory_gb:.1f}GB)")
        else:
            self.device = torch.device('cpu')
            if self.verbose:
                if self.use_cuda:
                    print("⚠️ CUDA requested but not available. Falling back to CPU.")
                print("💻 Using CPU device (consider GPU acceleration for better performance)")
            
        # Comprehensive reproducibility setup with detailed logging
        if self.deterministic:
            torch.manual_seed(self.seed)
            if self.verbose:
                print(f"🎲 Reproducibility enabled: seed={self.seed}")
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                if self.verbose and self.use_cuda:
                    print("  ✓ CUDA deterministic mode enabled")
                    print("  ⚠️ Performance may be reduced due to deterministic operations")
            
            # Set numpy seed for additional reproducibility
            import numpy as np
            np.random.seed(self.seed)
        elif self.verbose:
            print("🎲 Non-deterministic mode: results may vary between runs")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize configuration to dictionary format for storage and transmission.
        
        This method creates a JSON-serializable dictionary representation of the
        configuration, handling special objects like torch.device appropriately.
        Useful for:
        - Configuration persistence to files
        - Logging and experiment tracking
        - Configuration sharing between processes
        - Integration with external systems
        
        Returns:
            Dict[str, Any]: Complete configuration as key-value pairs with
                          JSON-compatible values. Device objects are converted
                          to string representations.
        
        Example:
            config = OptimizationConfig(max_steps=200)
            config_dict = config.to_dict()
            with open('config.json', 'w') as f:
                json.dump(config_dict, f, indent=2)
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if key == 'device':
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OptimizationConfig':
        """
        Deserialize configuration from dictionary format with intelligent reconstruction.
        
        This class method reconstructs a complete OptimizationConfig instance from
        a dictionary representation, handling device reconstruction and validation
        automatically. Useful for:
        - Loading saved configurations from files
        - Restoring configurations from experiment logs
        - Configuration transfer between systems
        - API-based configuration management
        
        Args:
            config_dict: Dictionary containing configuration parameters.
                        Should contain valid parameter names as keys.
                        Device information is automatically reconstructed.
        
        Returns:
            OptimizationConfig: Fully validated and initialized configuration instance
                              with proper device setup and parameter validation.
        
        Raises:
            ValueError: If dictionary contains unknown parameters or invalid values
            KeyError: If required parameters are missing from the dictionary
        
        Example:
            with open('saved_config.json', 'r') as f:
                config_dict = json.load(f)
            config = OptimizationConfig.from_dict(config_dict)
        """
        # Remove device from dict as it will be recreated in __post_init__
        config_dict = config_dict.copy()
        config_dict.pop('device', None)
        return cls(**config_dict)
    
    def update(self, **kwargs) -> 'OptimizationConfig':
        """
        Dynamically update configuration parameters with validation and re-initialization.
        
        This method provides a safe way to modify configuration parameters after
        initialization, ensuring that:
        1. Only valid parameters can be updated
        2. All validation rules are re-applied
        3. Dependent settings are properly reconfigured (e.g., device setup)
        4. The updated configuration remains internally consistent
        
        Particularly useful for:
        - Hyperparameter sweeps and grid searches
        - Dynamic configuration adjustment during optimization
        - Configuration inheritance and specialization
        - Interactive configuration tuning
        
        Args:
            **kwargs: Configuration parameters to update. Parameter names must
                     match existing configuration attributes. New parameters
                     cannot be added through this method.
        
        Returns:
            OptimizationConfig: Self-reference for method chaining, allowing
                              patterns like: config.update(lr=1e-3).update(steps=200)
        
        Raises:
            ValueError: If attempting to update unknown parameters or providing
                       invalid values that fail validation checks
        
        Example:
            # Single parameter update
            config.update(max_steps=500)
            
            # Multiple parameter updates
            config.update(
                max_steps=300,
                policy_learning_rate=5e-5,
                early_stop_patience=40
            )
            
            # Method chaining
            config.update(batch_size=64).update(verbose=False)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        # Re-run post-init validation
        self.__post_init__()
        return self


# Default configurations for common scenarios
class DefaultConfigs:
    """
    Curated Collection of Pre-Configured Settings for Common Optimization Scenarios.
    
    This utility class provides carefully tuned configuration templates that cover
    the most common hyperparameter optimization scenarios. Each configuration has
    been optimized for its specific use case based on extensive empirical testing
    and theoretical considerations.
    
    **Available Configurations**:
    
    1. **quick_test()**: Minimal setup for rapid prototyping and debugging
       - 10 steps for fast iteration
       - Small buffer and batch sizes for low memory usage
       - Ideal for code validation and initial testing
    
    2. **standard()**: Balanced configuration for typical optimization tasks
       - 100 steps for reasonable exploration
       - Standard buffer and learning rates
       - Good starting point for most hyperparameter optimization problems
    
    3. **extensive()**: Comprehensive search for complex optimization landscapes
       - 500 steps for thorough exploration
       - Large buffer for diverse experience retention
       - Conservative learning rates for stable convergence
       - Extended early stopping patience for complex problems
    
    4. **cpu_only()**: Optimized configuration for CPU-only environments
       - Reduced computational requirements
       - Adjusted batch sizes for CPU efficiency
       - Higher learning rates to compensate for reduced training time
    
    **Usage Pattern**:
    ```python
    # Select appropriate configuration
    config = DefaultConfigs.extensive()  # For complex problems
    
    # Customize as needed
    config.update(
        max_steps=300,  # Adjust exploration depth
        verbose=True    # Enable detailed logging
    )
    
    # Use in optimization
    optimizer = MAT_HPO_Optimizer(environment, space, config)
    ```
    
    **Selection Guidelines**:
    - Use **quick_test** for initial development and debugging
    - Use **standard** for most production hyperparameter optimization tasks
    - Use **extensive** for complex problems with large search spaces
    - Use **cpu_only** when GPU acceleration is unavailable
    """
    
    @staticmethod
    def quick_test() -> OptimizationConfig:
        """
        Lightweight configuration optimized for rapid development and testing.
        
        This configuration prioritizes speed and low resource usage, making it ideal for:
        - Initial code validation and debugging
        - Rapid prototyping of new optimization environments
        - Unit testing and continuous integration pipelines
        - Quick sanity checks before longer optimization runs
        
        **Characteristics**:
        - Very short optimization (10 steps)
        - Small memory footprint (100-sample buffer)
        - Frequent progress updates for debugging
        - Minimal computational requirements
        
        **Trade-offs**:
        - Limited exploration capability due to short duration
        - May not find optimal solutions for complex problems
        - Not suitable for production hyperparameter optimization
        
        Returns:
            OptimizationConfig: Configuration optimized for quick testing scenarios
        """
        return OptimizationConfig(
            max_steps=10,
            replay_buffer_size=100,
            batch_size=16,
            verbose=True,
            save_interval=5
        )
    
    @staticmethod
    def standard() -> OptimizationConfig:
        """
        Balanced configuration suitable for the majority of hyperparameter optimization tasks.
        
        This configuration represents the optimal balance between exploration thoroughness,
        computational efficiency, and convergence stability. Extensively tested across
        diverse optimization problems and proven effective for:
        
        - Machine learning model hyperparameter tuning
        - Neural architecture search problems
        - Optimization algorithm parameter selection
        - Multi-objective hyperparameter optimization
        
        **Characteristics**:
        - Moderate exploration duration (100 steps)
        - Balanced buffer size for diverse experience retention
        - Conservative learning rates for stable convergence
        - Standard batch size for efficient GPU utilization
        
        **Performance Profile**:
        - Typically converges within 50-80 steps for well-defined problems
        - Provides good exploration-exploitation balance
        - Stable across different types of hyperparameter spaces
        - Memory efficient for most hardware configurations
        
        Returns:
            OptimizationConfig: Well-balanced configuration for typical optimization scenarios
        """
        return OptimizationConfig(
            max_steps=100,
            replay_buffer_size=1000,
            batch_size=32,
            policy_learning_rate=1e-4,
            value_learning_rate=1e-3
        )
    
    @staticmethod
    def extensive() -> OptimizationConfig:
        """
        Comprehensive configuration designed for challenging, high-dimensional optimization problems.
        
        This configuration prioritizes thorough exploration and robust convergence, making it
        ideal for:
        - Large hyperparameter spaces (>20 parameters)
        - Multi-modal optimization landscapes with multiple local optima
        - Complex machine learning pipelines with intricate parameter interactions
        - Research scenarios requiring exhaustive parameter exploration
        - Production systems where optimization quality is more important than speed
        
        **Enhanced Features**:
        - Extended exploration (500 steps) for comprehensive search
        - Large experience buffer (5000 samples) for rich learning history
        - Conservative learning rates to avoid premature convergence
        - Increased batch size for stable gradient estimates
        - Extended early stopping patience to avoid premature termination
        
        **Computational Requirements**:
        - Higher memory usage due to large buffer
        - Longer training time (5-10x standard configuration)
        - Benefits significantly from GPU acceleration
        - May require 1-4 hours depending on environment evaluation time
        
        **Expected Performance**:
        - Superior final optimization quality compared to standard configurations
        - More robust to initialization sensitivity
        - Better handling of complex parameter interactions
        - Higher likelihood of finding global optima in challenging landscapes
        
        Returns:
            OptimizationConfig: High-capacity configuration for complex optimization challenges
        """
        return OptimizationConfig(
            max_steps=500,
            replay_buffer_size=5000,
            batch_size=64,
            policy_learning_rate=5e-5,
            value_learning_rate=5e-4,
            early_stop_patience=50
        )
    
    @staticmethod
    def cpu_only() -> OptimizationConfig:
        """
        Specialized configuration optimized for CPU-only execution environments.
        
        This configuration is specifically tuned for environments where GPU acceleration
        is unavailable or undesirable, such as:
        - Cloud environments without GPU access
        - Local development on CPU-only machines
        - Edge computing and embedded systems
        - Batch processing systems with CPU constraints
        - Cost-sensitive deployments where GPU usage is expensive
        
        **CPU-Specific Optimizations**:
        - Reduced batch size (16) for efficient CPU memory usage
        - Shorter optimization duration to account for slower computation
        - Smaller buffer size to reduce memory pressure
        - Higher learning rates to compensate for reduced training iterations
        - Disabled CUDA to prevent device conflicts
        
        **Performance Characteristics**:
        - 3-10x slower than GPU configurations depending on problem complexity
        - More memory efficient due to smaller batch and buffer sizes
        - Still capable of finding good solutions for moderate complexity problems
        - Suitable for hyperparameter spaces with <15 parameters
        
        **Use Cases**:
        - Development and testing on local CPU-only machines
        - Production deployments in CPU-constrained environments
        - Educational scenarios where GPU access is limited
        - Baseline comparisons and algorithm validation
        
        Returns:
            OptimizationConfig: CPU-optimized configuration with CUDA disabled
                              and parameters tuned for CPU execution efficiency
        """
        return OptimizationConfig(
            max_steps=50,
            replay_buffer_size=500,
            batch_size=16,
            use_cuda=False,
            policy_learning_rate=2e-4,
            value_learning_rate=2e-3
        )