"""
Hyperparameter Space Definition for MAT-HPO

This module provides a comprehensive framework for defining and managing 
hyperparameter search spaces in multi-agent optimization scenarios. It supports
complex parameter types, constraints, and validation with intelligent fallback
mechanisms and enhanced usability features.

Key Features:
- Multi-agent parameter partitioning with flexible assignment
- Support for diverse parameter types (int, float, categorical, boolean)
- Advanced constraint handling and validation
- Automatic bounds checking and type conversion
- Built-in sampling and grid search capabilities
- Integration with popular ML frameworks
- Checkpoint and serialization support

Example:
    >>> # Define search space for a neural network
    >>> space = HyperparameterSpace(
    ...     agent0_params=['class_weight_0', 'class_weight_1'],
    ...     agent1_params=['hidden_size', 'num_layers'],
    ...     agent2_params=['batch_size', 'learning_rate', 'optimizer_type'],
    ...     bounds={
    ...         'class_weight_0': (0.1, 5.0),
    ...         'class_weight_1': (0.1, 5.0),
    ...         'hidden_size': (64, 512),
    ...         'num_layers': (2, 8),
    ...         'batch_size': (16, 128),
    ...         'learning_rate': (1e-5, 1e-2),
    ...         'optimizer_type': ['adam', 'sgd', 'adamw']  # Categorical
    ...     },
    ...     param_types={
    ...         'class_weight_0': float,
    ...         'class_weight_1': float,
    ...         'hidden_size': int,
    ...         'num_layers': int,
    ...         'batch_size': int,
    ...         'learning_rate': float,
    ...         'optimizer_type': str
    ...     }
    ... )
"""

from typing import Dict, List, Tuple, Union, Any, Optional, Callable
import torch
import numpy as np
import json
import copy
import warnings
from enum import Enum
import itertools


class ParameterType(Enum):
    """Enumeration of supported parameter types."""
    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    LOG_UNIFORM = "log_uniform"  # For parameters like learning rate


class HyperparameterSpace:
    """
    Advanced hyperparameter search space for multi-agent optimization.
    
    This class provides a comprehensive framework for defining and managing
    hyperparameter search spaces across multiple optimization agents. Each agent
    can be responsible for different types of parameters, allowing for specialized
    optimization strategies.
    
    Key Features:
    - Multi-agent parameter partitioning with flexible assignment
    - Support for diverse parameter types (numerical, categorical, boolean)
    - Advanced constraint handling and validation
    - Automatic bounds checking and intelligent type conversion
    - Built-in sampling capabilities for testing and initialization
    - Integration with checkpointing and serialization
    - Extensive validation and error checking
    
    Typical Agent Assignments:
    - Agent 0: Problem-specific parameters (class weights, regularization, domain params)
    - Agent 1: Model architecture parameters (layers, units, network structure)
    - Agent 2: Training parameters (batch size, learning rate, optimization settings)
    
    Attributes:
        agent0_params (List[str]): Parameter names assigned to agent 0
        agent1_params (List[str]): Parameter names assigned to agent 1  
        agent2_params (List[str]): Parameter names assigned to agent 2
        bounds (Dict): Parameter bounds and constraints
        param_types (Dict): Parameter type specifications
        default_values (Dict): Default parameter values
        constraints (List): Custom constraint functions
        agent_dims (List[int]): Number of parameters per agent
    """
    
    def __init__(self,
                 agent0_params: Optional[List[str]] = None,
                 agent1_params: Optional[List[str]] = None, 
                 agent2_params: Optional[List[str]] = None,
                 bounds: Optional[Dict[str, Union[Tuple[float, float], List[str]]]] = None,
                 param_types: Optional[Dict[str, Union[type, str]]] = None,
                 default_values: Optional[Dict[str, Any]] = None,
                 constraints: Optional[List[Callable]] = None,
                 parameter_descriptions: Optional[Dict[str, str]] = None):
        """
        Initialize the hyperparameter search space.
        
        Args:
            agent0_params: List of parameter names for agent 0 (typically problem-specific)
            agent1_params: List of parameter names for agent 1 (typically architecture)
            agent2_params: List of parameter names for agent 2 (typically training)
            bounds: Dictionary mapping parameter names to bounds:
                   - For numerical: (min, max) tuple
                   - For categorical: List of valid values
                   - For boolean: (False, True) or just bool
            param_types: Dictionary mapping parameter names to types:
                        - int, float: Numerical parameters
                        - str: Categorical parameters
                        - bool: Boolean parameters
                        - 'log_uniform': Log-uniform distributed parameters
            default_values: Optional default values for parameters
            constraints: Optional list of constraint functions
            parameter_descriptions: Optional descriptions for documentation
            
        Example:
            >>> space = HyperparameterSpace(
            ...     agent0_params=['regularization', 'class_balance'],
            ...     agent1_params=['hidden_size', 'num_layers'],
            ...     agent2_params=['batch_size', 'learning_rate', 'optimizer'],
            ...     bounds={
            ...         'regularization': (0.0, 1.0),
            ...         'class_balance': [True, False],
            ...         'hidden_size': (64, 512),
            ...         'num_layers': (2, 8),
            ...         'batch_size': (16, 128),
            ...         'learning_rate': (1e-5, 1e-2),
            ...         'optimizer': ['adam', 'sgd', 'rmsprop']
            ...     },
            ...     param_types={
            ...         'regularization': float,
            ...         'class_balance': bool,
            ...         'hidden_size': int,
            ...         'num_layers': int,
            ...         'batch_size': int,
            ...         'learning_rate': 'log_uniform',
            ...         'optimizer': str
            ...     }
            ... )
        """
        # Support both legacy initialization and new dynamic approach
        if agent0_params is not None and agent1_params is not None and agent2_params is not None:
            # Legacy initialization with all parameters provided
            self._legacy_init(agent0_params, agent1_params, agent2_params, bounds, param_types, 
                            default_values, constraints, parameter_descriptions)
        else:
            # New dynamic initialization - parameters added via add_* methods
            self._dynamic_init(default_values, constraints, parameter_descriptions)

    def _legacy_init(self, agent0_params, agent1_params, agent2_params, bounds, param_types,
                    default_values, constraints, parameter_descriptions):
        """Legacy initialization method for backward compatibility."""
        # Validate input parameters
        if not all(isinstance(params, list) for params in [agent0_params, agent1_params, agent2_params]):
            raise TypeError("All agent parameter lists must be of type List[str]")
        
        self.agent0_params = agent0_params.copy()
        self.agent1_params = agent1_params.copy() 
        self.agent2_params = agent2_params.copy()
        
        # Check for parameter overlap
        all_params = agent0_params + agent1_params + agent2_params
        if len(all_params) != len(set(all_params)):
            duplicates = [p for p in all_params if all_params.count(p) > 1]
            raise ValueError(f"Duplicate parameters found: {duplicates}")
        
        # Validate that all parameters have bounds and types
        self._validate_configuration(all_params, bounds, param_types)
        
        self.bounds = copy.deepcopy(bounds)
        self.param_types = param_types.copy()
        self.default_values = default_values.copy() if default_values else {}
        self.constraints = constraints or []
        self.parameter_descriptions = parameter_descriptions or {}
        
        # Compute dimensions for each agent
        self.agent_dims = [
            len(agent0_params),
            len(agent1_params), 
            len(agent2_params)
        ]
        
        # Process and categorize parameters
        self._categorize_parameters()
        
        # Create bounds tensors for each agent (for numerical parameters)
        self._create_bounds_tensors()
        
        # Validation flag
        self._is_validated = True
        
        # Performance tracking
        self._translation_cache = {}
        
    def _dynamic_init(self, default_values, constraints, parameter_descriptions):
        """New dynamic initialization for add_* methods."""
        # Initialize empty collections
        self.parameters = {}  # All parameters with their config
        self.agent0_params = []
        self.agent1_params = []
        self.agent2_params = []
        self.bounds = {}
        self.param_types = {}
        self.default_values = default_values or {}
        self.constraints = constraints or []
        self.parameter_descriptions = parameter_descriptions or {}
        
        # Initialize parameter categorization dictionaries
        self.numerical_params = {}
        self.discrete_params = {}
        self.boolean_params = {}
        
        # Initialize agent dimensions
        self.agent_dims = (0, 0, 0)
        
        # Initialize state
        self._is_validated = True
        self._translation_cache = {}
        self.device = None
        
        # Initialize tensors (will be populated when parameters are added)
        self.hL_tensors = []
        self.hU_tensors = []
        
    def add_continuous(self, name: str, min_val: float, max_val: float, 
                      agent: int, log_scale: bool = False, 
                      description: str = None) -> 'HyperparameterSpace':
        """
        Add a continuous hyperparameter to the search space.
        
        Args:
            name: Parameter name
            min_val: Minimum value
            max_val: Maximum value  
            agent: Agent ID (0, 1, or 2)
            log_scale: Whether to use log-uniform distribution
            description: Optional description
            
        Returns:
            Self for method chaining
        """
        if agent not in [0, 1, 2]:
            raise ValueError("Agent must be 0, 1, or 2")
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")
        if name in self.parameters:
            raise ValueError(f"Parameter {name} already exists")
            
        param_type = 'log_uniform' if log_scale else 'continuous'
        
        self.parameters[name] = {
            'type': param_type,
            'bounds': (min_val, max_val),
            'agent': agent,
            'description': description
        }
        
        # Add to appropriate agent list
        if agent == 0:
            self.agent0_params.append(name)
        elif agent == 1:
            self.agent1_params.append(name)
        else:
            self.agent2_params.append(name)
            
        # Update legacy attributes for backward compatibility
        self.bounds[name] = (min_val, max_val)
        self.param_types[name] = 'continuous' if not log_scale else 'log_uniform'
            
        # Update agent dimensions
        self.agent_dims = (len(self.agent0_params), len(self.agent1_params), len(self.agent2_params))
        
        return self
    
    def add_discrete(self, name: str, choices: List, agent: int,
                    description: str = None) -> 'HyperparameterSpace':
        """
        Add a discrete hyperparameter to the search space.
        
        Args:
            name: Parameter name
            choices: List of valid choices
            agent: Agent ID (0, 1, or 2)
            description: Optional description
            
        Returns:
            Self for method chaining
        """
        if agent not in [0, 1, 2]:
            raise ValueError("Agent must be 0, 1, or 2")
        if not choices:
            raise ValueError("Choices list cannot be empty")
        if name in self.parameters:
            raise ValueError(f"Parameter {name} already exists")
            
        self.parameters[name] = {
            'type': 'discrete',
            'choices': choices,
            'agent': agent,
            'description': description
        }
        
        # Add to appropriate agent list
        if agent == 0:
            self.agent0_params.append(name)
        elif agent == 1:
            self.agent1_params.append(name)  
        else:
            self.agent2_params.append(name)
            
        # Update legacy attributes for backward compatibility
        self.bounds[name] = choices  # For discrete params, bounds stores choices
        self.param_types[name] = 'discrete'
            
        # Update agent dimensions
        self.agent_dims = (len(self.agent0_params), len(self.agent1_params), len(self.agent2_params))
        
        return self
    
    def add_boolean(self, name: str, agent: int,
                   description: str = None) -> 'HyperparameterSpace':
        """
        Add a boolean hyperparameter to the search space.
        
        Args:
            name: Parameter name
            agent: Agent ID (0, 1, or 2)
            description: Optional description
            
        Returns:
            Self for method chaining
        """
        if agent not in [0, 1, 2]:
            raise ValueError("Agent must be 0, 1, or 2")
        if name in self.parameters:
            raise ValueError(f"Parameter {name} already exists")
            
        self.parameters[name] = {
            'type': 'boolean',
            'agent': agent,
            'description': description
        }
        
        # Add to appropriate agent list
        if agent == 0:
            self.agent0_params.append(name)
        elif agent == 1:
            self.agent1_params.append(name)
        else:
            self.agent2_params.append(name)
            
        # Update agent dimensions  
        self.agent_dims = (len(self.agent0_params), len(self.agent1_params), len(self.agent2_params))
        
        return self
        
    def _validate_configuration(self, all_params: List[str], 
                               bounds: Dict, param_types: Dict):
        """Validate the hyperparameter space configuration."""
        # Check that all parameters have bounds and types
        for param in all_params:
            if param not in bounds:
                raise ValueError(f"Missing bounds for parameter: {param}")
            if param not in param_types:
                raise ValueError(f"Missing type for parameter: {param}")
            
            # Validate bounds format
            param_bounds = bounds[param]
            param_type = param_types[param]
            
            if param_type in [int, float, 'log_uniform']:
                if not (isinstance(param_bounds, tuple) and len(param_bounds) == 2):
                    raise ValueError(f"Numerical parameter {param} requires (min, max) bounds tuple")
                if param_bounds[0] >= param_bounds[1]:
                    raise ValueError(f"Invalid bounds for {param}: min must be < max")
            elif param_type == bool:
                if not isinstance(param_bounds, (tuple, list)) or len(param_bounds) != 2:
                    # Auto-set boolean bounds
                    bounds[param] = (False, True)
            elif param_type == str:
                if not isinstance(param_bounds, (list, tuple)):
                    raise ValueError(f"Categorical parameter {param} requires list of valid values")
                if len(param_bounds) == 0:
                    raise ValueError(f"Categorical parameter {param} must have at least one valid value")
    
    def _categorize_parameters(self):
        """Categorize parameters by type for optimized processing."""
        self.numerical_params = {}
        self.categorical_params = {}
        self.boolean_params = {}
        self.log_uniform_params = {}
        
        all_params = self.agent0_params + self.agent1_params + self.agent2_params
        
        for param in all_params:
            param_type = self.param_types[param]
            bounds = self.bounds[param]
            
            if param_type in [int, float]:
                self.numerical_params[param] = {'type': param_type, 'bounds': bounds}
            elif param_type == 'log_uniform':
                self.log_uniform_params[param] = {'bounds': bounds}
            elif param_type == str:
                self.categorical_params[param] = {'values': bounds}
            elif param_type == bool:
                self.boolean_params[param] = {'values': bounds}
    
    def _create_bounds_tensors(self, device=None):
        """Create lower and upper bound tensors for numerical parameters of each agent."""
        self.hL_tensors = []
        self.hU_tensors = []
        self.numerical_indices = []  # Track which indices are numerical
        
        for agent_params in [self.agent0_params, self.agent1_params, self.agent2_params]:
            lower_bounds = []
            upper_bounds = []
            numerical_idx = []
            
            for i, param in enumerate(agent_params):
                param_type = self.param_types[param]
                
                if param_type in [int, float, 'log_uniform']:
                    bounds = self.bounds[param]
                    if param_type == 'log_uniform':
                        # Convert to log space for uniform sampling
                        lower_bounds.append(np.log(bounds[0]))
                        upper_bounds.append(np.log(bounds[1]))
                    else:
                        lower_bounds.append(bounds[0])
                        upper_bounds.append(bounds[1])
                    numerical_idx.append(i)
                else:
                    # For categorical/boolean, we'll handle separately
                    lower_bounds.append(0.0)
                    upper_bounds.append(1.0)
            
            # Convert to tensors
            hL = torch.tensor(lower_bounds, dtype=torch.float32)
            hU = torch.tensor(upper_bounds, dtype=torch.float32)
            
            if device is not None:
                hL = hL.to(device)
                hU = hU.to(device)
            
            self.hL_tensors.append(hL)
            self.hU_tensors.append(hU)
            self.numerical_indices.append(numerical_idx)
    
    def to_device(self, device: Union[str, torch.device]):
        """Move bounds tensors to specified device."""
        if isinstance(device, str):
            device = torch.device(device)
            
        for i in range(len(self.hL_tensors)):
            self.hL_tensors[i] = self.hL_tensors[i].to(device)
            self.hU_tensors[i] = self.hU_tensors[i].to(device)
        return self
    
    def get_agent_bounds(self, agent_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the bounds tensors for a specific agent.
        
        Args:
            agent_id: Agent ID (0, 1, or 2)
            
        Returns:
            Tuple of (lower_bounds, upper_bounds) tensors
            
        Raises:
            ValueError: If agent_id is not valid
        """
        if agent_id not in [0, 1, 2]:
            raise ValueError(f"Invalid agent_id: {agent_id}. Must be 0, 1, or 2.")
            
        return self.hL_tensors[agent_id], self.hU_tensors[agent_id]
    
    def get_agent_params(self, agent_id: int) -> List[str]:
        """
        Get the parameter names for a specific agent.
        
        Args:
            agent_id: Agent ID (0, 1, or 2)
            
        Returns:
            List of parameter names for the agent
            
        Raises:
            ValueError: If agent_id is not valid
        """
        if agent_id == 0:
            return self.agent0_params.copy()
        elif agent_id == 1:
            return self.agent1_params.copy()
        elif agent_id == 2:
            return self.agent2_params.copy()
        else:
            raise ValueError(f"Invalid agent_id: {agent_id}. Must be 0, 1, or 2.")
    
    def translate_actions(self, actions: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Translate normalized actions from agents to actual hyperparameter values.
        
        This method converts the normalized action outputs from each agent (typically
        in the range [-1, 1]) to actual hyperparameter values according to their
        defined types and bounds.
        
        Args:
            actions: List of action tensors from each agent (normalized to [-1, 1])
                    Must contain exactly 3 tensors for agents 0, 1, 2
            
        Returns:
            Dictionary mapping parameter names to their actual values
            
        Raises:
            ValueError: If actions list doesn't have exactly 3 tensors
            
        Example:
            >>> actions = [torch.tensor([0.5, -0.2]), torch.tensor([0.8]), torch.tensor([0.1, -0.9])]
            >>> hyperparams = space.translate_actions(actions)
            >>> print(hyperparams)
            {'class_weight_0': 3.25, 'class_weight_1': 1.4, 'hidden_size': 256, ...}
        """
        if len(actions) != 3:
            raise ValueError(f"Expected 3 action tensors (one per agent), got {len(actions)}")
        
        hyperparams = {}
        
        for agent_id, action in enumerate(actions):
            agent_params = self.get_agent_params(agent_id)
            
            if len(action) != len(agent_params):
                raise ValueError(f"Action tensor for agent {agent_id} has wrong size: "
                               f"expected {len(agent_params)}, got {len(action)}")
            
            # Process each parameter for this agent
            for i, param_name in enumerate(agent_params):
                action_value = action[i].item()
                param_type = self.param_types[param_name]
                bounds = self.bounds[param_name]
                
                if param_type in [int, float] or param_type == 'continuous':
                    # Convert from [-1, 1] to [min, max]
                    min_val, max_val = bounds
                    scaled_value = min_val + (max_val - min_val) * (1 + action_value) / 2
                    
                    # Apply bounds clipping
                    scaled_value = max(min(scaled_value, max_val), min_val)
                    
                    # Apply type conversion
                    if param_type == int:
                        value = int(round(scaled_value))
                    else:
                        value = float(scaled_value)
                        
                elif param_type == 'log_uniform':
                    # Convert from [-1, 1] to log space, then exponentiate
                    min_log, max_log = np.log(bounds[0]), np.log(bounds[1])
                    log_value = min_log + (max_log - min_log) * (1 + action_value) / 2
                    value = float(np.exp(log_value))
                    
                    # Ensure within original bounds
                    value = max(min(value, bounds[1]), bounds[0])
                    
                elif param_type == 'discrete' or param_type == str:
                    # Convert to categorical choice
                    choices = bounds
                    # Map [-1, 1] to [0, len(choices)-1]
                    choice_idx = int(np.clip((1 + action_value) / 2 * len(choices), 0, len(choices) - 1))
                    value = choices[choice_idx]
                    
                elif param_type == bool:
                    # Convert to boolean
                    value = action_value > 0.0
                    
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type}")
                
                hyperparams[param_name] = value
        
        # Apply constraints if any
        hyperparams = self._apply_constraints(hyperparams)
        
        return hyperparams
    
    def _apply_constraints(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom constraints to hyperparameters."""
        if not self.constraints:
            return hyperparams
        
        for constraint_func in self.constraints:
            try:
                hyperparams = constraint_func(hyperparams)
            except Exception as e:
                warnings.warn(f"Constraint function failed: {e}")
        
        return hyperparams
    
    def sample_random(self, n_samples: int = 1) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Sample random hyperparameter configurations.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Single dict if n_samples=1, otherwise list of dicts
        """
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            
            for param in self.agent0_params + self.agent1_params + self.agent2_params:
                param_type = self.param_types[param]
                bounds = self.bounds[param]
                
                if param_type == int:
                    sample[param] = np.random.randint(bounds[0], bounds[1] + 1)
                elif param_type == float:
                    sample[param] = np.random.uniform(bounds[0], bounds[1])
                elif param_type == 'log_uniform':
                    log_val = np.random.uniform(np.log(bounds[0]), np.log(bounds[1]))
                    sample[param] = float(np.exp(log_val))
                elif param_type == str:
                    sample[param] = np.random.choice(bounds)
                elif param_type == bool:
                    sample[param] = np.random.choice([False, True])
            
            # Apply constraints
            sample = self._apply_constraints(sample)
            samples.append(sample)
        
        return samples[0] if n_samples == 1 else samples
    
    def grid_search_configs(self, resolution: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """
        Generate grid search configurations.
        
        Args:
            resolution: Number of values to test for each parameter
                       If None, uses reasonable defaults
            
        Returns:
            List of all possible hyperparameter combinations
        """
        if resolution is None:
            resolution = {}
        
        param_grids = {}
        all_params = self.agent0_params + self.agent1_params + self.agent2_params
        
        for param in all_params:
            param_type = self.param_types[param]
            bounds = self.bounds[param]
            n_values = resolution.get(param, 3)  # Default: 3 values per parameter
            
            if param_type == int:
                param_grids[param] = list(range(bounds[0], bounds[1] + 1, 
                                              max(1, (bounds[1] - bounds[0]) // n_values)))
            elif param_type == float:
                param_grids[param] = list(np.linspace(bounds[0], bounds[1], n_values))
            elif param_type == 'log_uniform':
                log_values = np.linspace(np.log(bounds[0]), np.log(bounds[1]), n_values)
                param_grids[param] = [float(np.exp(v)) for v in log_values]
            elif param_type == str:
                param_grids[param] = bounds
            elif param_type == bool:
                param_grids[param] = [False, True]
        
        # Generate all combinations
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        configs = []
        for combination in itertools.product(*param_values):
            config = dict(zip(param_names, combination))
            config = self._apply_constraints(config)
            configs.append(config)
        
        return configs
    
    def get_default_hyperparams(self) -> Dict[str, Any]:
        """
        Get default hyperparameter values.
        
        Returns:
            Dictionary of default hyperparameter values
            
        If no default is specified for a parameter, uses:
        - Middle of range for numerical parameters
        - First option for categorical parameters  
        - False for boolean parameters
        """
        defaults = {}
        all_params = self.agent0_params + self.agent1_params + self.agent2_params
        
        for param in all_params:
            if param in self.default_values:
                defaults[param] = self.default_values[param]
            else:
                param_type = self.param_types[param]
                bounds = self.bounds[param]
                
                if param_type == int:
                    defaults[param] = int((bounds[0] + bounds[1]) // 2)
                elif param_type == float:
                    defaults[param] = float((bounds[0] + bounds[1]) / 2)
                elif param_type == 'log_uniform':
                    log_mid = (np.log(bounds[0]) + np.log(bounds[1])) / 2
                    defaults[param] = float(np.exp(log_mid))
                elif param_type == str:
                    defaults[param] = bounds[0]  # First option
                elif param_type == bool:
                    defaults[param] = False
                    
        return defaults
    
    def validate_hyperparams(self, hyperparams: Dict[str, Any], 
                           strict: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate that hyperparameters are within bounds and correct types.
        
        Args:
            hyperparams: Dictionary of hyperparameter values to validate
            strict: If True, requires all parameters to be present
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        all_params = self.agent0_params + self.agent1_params + self.agent2_params
        
        # Check for missing parameters
        for param in all_params:
            if param not in hyperparams:
                if strict:
                    errors.append(f"Missing required parameter: {param}")
                continue
                
            value = hyperparams[param]
            param_type = self.param_types[param]
            bounds = self.bounds[param]
            
            # Check type
            expected_types = {
                int: int,
                float: (int, float),  # Allow int for float params
                'log_uniform': (int, float),
                str: str,
                bool: bool
            }
            
            if param_type in expected_types:
                expected = expected_types[param_type]
                if not isinstance(value, expected):
                    errors.append(f"Wrong type for {param}: expected {expected}, got {type(value)}")
                    continue
            
            # Check bounds/constraints
            if param_type in [int, float, 'log_uniform']:
                if not (bounds[0] <= value <= bounds[1]):
                    errors.append(f"Value for {param} out of bounds: {value} not in [{bounds[0]}, {bounds[1]}]")
            elif param_type == str:
                if value not in bounds:
                    errors.append(f"Invalid value for {param}: {value} not in {bounds}")
            elif param_type == bool:
                if value not in [True, False]:
                    errors.append(f"Invalid boolean value for {param}: {value}")
        
        return len(errors) == 0, errors
    
    def add_constraint(self, constraint_func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Add a custom constraint function.
        
        Args:
            constraint_func: Function that takes hyperparams dict and returns modified dict
        """
        self.constraints.append(constraint_func)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert hyperparameter space to dictionary for serialization.
        
        Returns:
            Dictionary representation of the hyperparameter space
        """
        return {
            'agent0_params': self.agent0_params,
            'agent1_params': self.agent1_params,
            'agent2_params': self.agent2_params,
            'bounds': self.bounds,
            'param_types': {k: str(v) for k, v in self.param_types.items()},
            'default_values': self.default_values,
            'parameter_descriptions': self.parameter_descriptions,
            'agent_dims': self.agent_dims
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HyperparameterSpace':
        """
        Create hyperparameter space from dictionary.
        
        Args:
            config_dict: Dictionary representation (from to_dict())
            
        Returns:
            HyperparameterSpace instance
        """
        # Convert string types back to actual types
        param_types = {}
        for k, v in config_dict['param_types'].items():
            if v == 'int':
                param_types[k] = int
            elif v == 'float':
                param_types[k] = float
            elif v == 'str':
                param_types[k] = str
            elif v == 'bool':
                param_types[k] = bool
            elif v == 'log_uniform':
                param_types[k] = 'log_uniform'
        
        return cls(
            agent0_params=config_dict['agent0_params'],
            agent1_params=config_dict['agent1_params'],
            agent2_params=config_dict['agent2_params'],
            bounds=config_dict['bounds'],
            param_types=param_types,
            default_values=config_dict.get('default_values', {}),
            parameter_descriptions=config_dict.get('parameter_descriptions', {})
        )
    
    def save_to_file(self, filepath: str):
        """Save hyperparameter space to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'HyperparameterSpace':
        """Load hyperparameter space from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def get_parameter_info(self, param_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific parameter.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Dictionary with parameter information
        """
        if param_name not in (self.agent0_params + self.agent1_params + self.agent2_params):
            raise ValueError(f"Parameter {param_name} not found in hyperparameter space")
        
        # Determine which agent owns this parameter
        agent_id = None
        if param_name in self.agent0_params:
            agent_id = 0
        elif param_name in self.agent1_params:
            agent_id = 1
        elif param_name in self.agent2_params:
            agent_id = 2
        
        return {
            'name': param_name,
            'agent_id': agent_id,
            'type': self.param_types[param_name],
            'bounds': self.bounds[param_name],
            'default': self.default_values.get(param_name, 'auto'),
            'description': self.parameter_descriptions.get(param_name, 'No description')
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the hyperparameter space.
        
        Returns:
            Dictionary with detailed summary information
        """
        all_params = self.agent0_params + self.agent1_params + self.agent2_params
        
        summary = {
            'total_parameters': len(all_params),
            'agent_distribution': {
                'agent_0': len(self.agent0_params),
                'agent_1': len(self.agent1_params), 
                'agent_2': len(self.agent2_params)
            }
        }
        
        # Add parameter types info if available (legacy initialization)
        if hasattr(self, 'param_types') and self.param_types:
            summary['parameter_types'] = {
                'numerical': len([p for p in all_params if self.param_types.get(p) in [int, float, 'log_uniform']]),
                'categorical': len([p for p in all_params if self.param_types.get(p) == str]),
                'boolean': len([p for p in all_params if self.param_types.get(p) == bool])
            }
        elif hasattr(self, 'parameters') and self.parameters:
            # New dynamic initialization
            param_types_count = {'numerical': 0, 'categorical': 0, 'boolean': 0}
            for param_info in self.parameters.values():
                param_type = param_info.get('type', 'continuous')
                if param_type in ['continuous', 'log_uniform']:
                    param_types_count['numerical'] += 1
                elif param_type == 'discrete':
                    param_types_count['categorical'] += 1
                elif param_type == 'boolean':
                    param_types_count['boolean'] += 1
            summary['parameter_types'] = param_types_count
        
        # Add optional info if available
        if hasattr(self, 'default_values'):
            summary['has_defaults'] = len(self.default_values)
        if hasattr(self, 'constraints'):
            summary['has_constraints'] = len(self.constraints)
        if hasattr(self, 'parameter_descriptions'):
            summary['has_descriptions'] = len(self.parameter_descriptions)
        
        return summary
    
    def __str__(self) -> str:
        """String representation of the hyperparameter space."""
        summary = self.get_summary()
        
        result = f"MAT-HPO Hyperparameter Space\n"
        result += f"{'=' * 35}\n"
        result += f"Total Parameters: {summary['total_parameters']}\n"
        result += f"\nAgent Distribution:\n"
        result += f"  Agent 0: {summary['agent_distribution']['agent_0']} parameters\n"
        result += f"  Agent 1: {summary['agent_distribution']['agent_1']} parameters\n" 
        result += f"  Agent 2: {summary['agent_distribution']['agent_2']} parameters\n"
        result += f"\nParameter Types:\n"
        result += f"  Numerical: {summary['parameter_types']['numerical']}\n"
        result += f"  Categorical: {summary['parameter_types']['categorical']}\n"
        result += f"  Boolean: {summary['parameter_types']['boolean']}\n"
        
        if summary['has_constraints'] > 0:
            result += f"\nConstraints: {summary['has_constraints']} custom constraint(s)\n"
        
        return result
    
    def __repr__(self) -> str:
        """Repr representation of the hyperparameter space."""
        return f"HyperparameterSpace(total_params={len(self.agent0_params + self.agent1_params + self.agent2_params)})"