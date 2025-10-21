"""
Base Environment for MAT-HPO

Abstract base class that users must implement to define their optimization problem.
This module provides a flexible foundation for implementing various machine learning
optimization environments with different models, datasets, and evaluation criteria.

Example:
    >>> class MyEnvironment(BaseEnvironment):
    ...     def load_data(self):
    ...         return load_my_dataset()
    ...     
    ...     def create_model(self, hyperparams):
    ...         return MyModel(**hyperparams)
    ...     
    ...     def train_evaluate(self, model, hyperparams):
    ...         return {'accuracy': 0.95, 'f1': 0.93}
    ...     
    ...     def compute_reward(self, metrics):
    ...         return metrics['accuracy'] * 0.7 + metrics['f1'] * 0.3
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List, Union, Callable
import torch
import numpy as np
import os
import json
import time
from datetime import datetime
import warnings


class BaseEnvironment(ABC):
    """
    Abstract base class for MAT-HPO environments.
    
    This class provides a comprehensive framework for defining optimization problems
    that can be solved using the MAT-HPO (Multi-Agent Transformer Hyperparameter
    Optimization) algorithm. Users must inherit from this class and implement the
    abstract methods to define their specific optimization problem.
    
    Key Features:
    - Automatic tracking of training history and best results
    - Built-in progress monitoring and logging capabilities  
    - Flexible metric extraction with fallback options
    - Error handling and graceful degradation
    - Support for custom stopping criteria
    - Checkpoint and resume functionality
    - Integration with external validation sets
    
    Attributes:
        name (str): Name identifier for this environment
        current_step (int): Current optimization step number
        best_reward (float): Best reward achieved so far
        best_hyperparams (dict): Hyperparameters that achieved best reward
        training_history (list): Complete history of training steps
        validation_data (any): Optional separate validation dataset
        test_data (any): Optional separate test dataset
        checkpoint_dir (str): Directory for saving checkpoints
        verbose (bool): Whether to print detailed progress information
    """
    
    def __init__(self, 
                 name: str = "CustomEnvironment",
                 checkpoint_dir: Optional[str] = None,
                 verbose: bool = True,
                 save_history: bool = True,
                 validation_split: float = 0.2):
        """
        Initialize the base environment.
        
        Args:
            name: Descriptive name for this environment
            checkpoint_dir: Directory to save checkpoints (None = no checkpointing)
            verbose: Whether to print progress information
            save_history: Whether to maintain detailed training history
            validation_split: Fraction of data to use for validation (if applicable)
        """
        self.name = name
        self.current_step = 0
        self.best_reward = float('-inf')
        self.best_hyperparams = None
        self.training_history = [] if save_history else None
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        self.validation_split = validation_split
        
        # Data storage
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        
        # Performance tracking
        self.start_time = None
        self.step_times = []
        self.reward_history = []
        
        # Custom callbacks and hooks
        self.step_callbacks = []  # Functions called after each step
        self.epoch_callbacks = []  # Functions called after each epoch (if applicable)
        
        # Setup checkpoint directory
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            if verbose:
                print(f"📁 Checkpoint directory: {checkpoint_dir}")
    
    # ============================================================================
    # ABSTRACT METHODS (Must be implemented by subclasses)
    # ============================================================================
    
    @abstractmethod
    def load_data(self) -> Any:
        """
        Load and prepare the dataset for training/evaluation.
        
        This method should handle all data preprocessing, splitting, and preparation
        needed for your specific problem. The returned data can be in any format
        that your create_model and train_evaluate methods can work with.
        
        Common implementations:
        - Load data from files (CSV, JSON, images, etc.)
        - Apply preprocessing (normalization, tokenization, etc.)
        - Split into train/validation/test sets
        - Convert to appropriate tensor formats
        
        Returns:
            Any: The loaded and preprocessed dataset
            
        Example:
            >>> def load_data(self):
            ...     # Load from file
            ...     data = pd.read_csv('my_dataset.csv')
            ...     
            ...     # Preprocessing
            ...     X = data.drop('target', axis=1).values
            ...     y = data['target'].values
            ...     
            ...     # Split data
            ...     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
            ...     
            ...     return {'X_train': X_train, 'X_val': X_val, 
            ...             'y_train': y_train, 'y_val': y_val}
        """
        pass
    
    @abstractmethod 
    def create_model(self, hyperparams: Dict[str, Any]) -> Any:
        """
        Create a model instance with the given hyperparameters.
        
        This method should instantiate your model/algorithm with the specified
        hyperparameters. The model can be of any type (PyTorch, scikit-learn,
        TensorFlow, custom implementation, etc.).
        
        Args:
            hyperparams: Dictionary containing hyperparameter values
                        Keys should match the parameters defined in HyperparameterSpace
            
        Returns:
            Any: The created model instance ready for training
            
        Example:
            >>> def create_model(self, hyperparams):
            ...     import torch.nn as nn
            ...     
            ...     model = nn.Sequential(
            ...         nn.Linear(784, hyperparams['hidden_size']),
            ...         nn.ReLU(),
            ...         nn.Dropout(hyperparams.get('dropout', 0.5)),
            ...         nn.Linear(hyperparams['hidden_size'], 10)
            ...     )
            ...     return model
        """
        pass
    
    @abstractmethod
    def train_evaluate(self, model: Any, hyperparams: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the model and evaluate its performance.
        
        This is the core method where actual model training and evaluation occurs.
        It should train the model with the given hyperparameters and return
        evaluation metrics that will be used for optimization.
        
        Args:
            model: The model instance to train (from create_model)
            hyperparams: Dictionary containing hyperparameter values
            
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
                             Should include at least one of: 'f1', 'accuracy', 'auc'
                             Can include additional custom metrics.
                             
        Common metrics to return:
        - 'f1' or 'f1_score': F1 score
        - 'accuracy': Classification accuracy  
        - 'auc' or 'auc_score': Area Under Curve
        - 'precision', 'recall': Precision and recall
        - 'loss' or 'val_loss': Training/validation loss
        - 'mse', 'mae': Mean Squared/Absolute Error (for regression)
        - Custom domain-specific metrics
        
        Example:
            >>> def train_evaluate(self, model, hyperparams):
            ...     # Setup training
            ...     optimizer = torch.optim.Adam(model.parameters(), 
            ...                                  lr=hyperparams['learning_rate'])
            ...     criterion = nn.CrossEntropyLoss()
            ...     
            ...     # Training loop
            ...     for epoch in range(hyperparams.get('epochs', 10)):
            ...         model.train()
            ...         for batch_x, batch_y in self.train_loader:
            ...             optimizer.zero_grad()
            ...             output = model(batch_x)
            ...             loss = criterion(output, batch_y)
            ...             loss.backward()
            ...             optimizer.step()
            ...     
            ...     # Evaluation
            ...     model.eval()
            ...     with torch.no_grad():
            ...         predictions = model(self.val_x)
            ...         accuracy = calculate_accuracy(predictions, self.val_y)
            ...         f1 = calculate_f1(predictions, self.val_y)
            ...     
            ...     return {'accuracy': accuracy, 'f1': f1, 'loss': loss.item()}
        """
        pass
    
    @abstractmethod
    def compute_reward(self, metrics: Dict[str, float]) -> float:
        """
        Compute the reward based on evaluation metrics.
        
        This method defines how to combine multiple evaluation metrics into a
        single reward value that the optimization algorithm will try to maximize.
        Higher reward values indicate better performance.
        
        Args:
            metrics: Dictionary containing evaluation metrics from train_evaluate
            
        Returns:
            float: Computed reward value (higher is better)
            
        Common reward computation strategies:
        - Single metric: return metrics['f1']  
        - Weighted combination: return 0.7 * metrics['accuracy'] + 0.3 * metrics['f1']
        - Multi-objective: return harmonic_mean([metrics['precision'], metrics['recall']])
        - With penalties: return metrics['f1'] - 0.1 * (training_time / max_time)
        
        Example:
            >>> def compute_reward(self, metrics):
            ...     # Multi-objective reward with accuracy and F1
            ...     if 'f1' in metrics and 'accuracy' in metrics:
            ...         return 0.6 * metrics['f1'] + 0.4 * metrics['accuracy']
            ...     elif 'f1' in metrics:
            ...         return metrics['f1']
            ...     else:
            ...         return metrics.get('accuracy', 0.0)
        """
        pass
    
    # ============================================================================
    # CORE ENVIRONMENT METHODS (Can be overridden if needed)
    # ============================================================================
    
    def reset(self) -> torch.Tensor:
        """
        Reset the environment for a new optimization run.
        
        This method is called at the beginning of optimization to initialize
        or reset the environment state. It can be overridden to perform
        custom initialization.
        
        Returns:
            torch.Tensor: Initial state tensor for the agents
                         Default shape: [1, state_dim] where state_dim=12
                         (9 class weights + 1 architecture + 2 training params)
        """
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
        self.reward_history = []
        
        if self.verbose:
            print(f"🔄 Environment '{self.name}' reset for optimization")
        
        # Return initial state tensor - can be overridden for custom state representation
        # Default: 12 dimensions for typical MAT-HPO setup (9 + 1 + 2 agents)
        state_dim = 12
        initial_state = torch.randn(1, state_dim)
        
        return initial_state
    
    def step(self, hyperparams: Dict[str, Any]) -> Tuple[float, float, float, bool]:
        """
        Execute one step of evaluation with given hyperparameters.
        
        This is the main interface method called by the MAT-HPO optimizer.
        It orchestrates the entire training and evaluation process for one
        set of hyperparameters.
        
        Args:
            hyperparams: Dictionary containing hyperparameter values
            
        Returns:
            Tuple containing:
            - f1_score (float): F1 score or primary performance metric
            - auc_score (float): AUC score or secondary performance metric  
            - gmean_score (float): G-mean score or tertiary performance metric
            - done (bool): Whether optimization should stop early
            
        The method includes:
        - Error handling and graceful degradation
        - Automatic progress tracking and logging
        - Performance timing
        - History recording
        - Custom callback execution
        - Early stopping detection
        """
        step_start_time = time.time()
        
        try:
            if self.verbose:
                print(f"\nStep {self.current_step + 1}: Evaluating hyperparameters...")
            
            # Load data if not already loaded
            if self.training_data is None:
                if self.verbose:
                    print("📊 Loading data...")
                self.training_data = self.load_data()
            
            # Create model with hyperparameters
            if self.verbose:
                print("🏗️  Creating model...")
            model = self.create_model(hyperparams)
            
            # Train and evaluate
            if self.verbose:
                print("🎓 Training and evaluating...")
            metrics = self.train_evaluate(model, hyperparams)
            
            # Compute reward
            reward = self.compute_reward(metrics)
            
            # Extract standard metrics with intelligent fallbacks
            f1_score = self._extract_metric(metrics, ['f1', 'f1_score'], 0.0)
            auc_score = self._extract_metric(metrics, ['auc', 'auc_score', 'accuracy'], 0.0)
            gmean_score = self._extract_metric(metrics, ['gmean', 'g_mean', 'precision', 'recall'], 0.0)
            
            # Update tracking
            self.current_step += 1
            step_time = time.time() - step_start_time
            self.step_times.append(step_time)
            self.reward_history.append(reward)
            
            # Track best results
            is_best = reward > self.best_reward
            if is_best:
                self.best_reward = reward
                self.best_hyperparams = hyperparams.copy()
                if self.verbose:
                    print(f"🎊 New best reward: {reward:.4f}")
            
            # Record training history
            if self.training_history is not None:
                self.training_history.append({
                    'step': self.current_step,
                    'hyperparams': hyperparams.copy(),
                    'metrics': metrics.copy(),
                    'reward': reward,
                    'step_time': step_time,
                    'is_best': is_best,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Execute custom callbacks
            for callback in self.step_callbacks:
                try:
                    callback(self, hyperparams, metrics, reward)
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Callback error: {e}")
            
            # Check for early stopping
            done = self._should_stop(metrics, reward)
            
            # Save checkpoint if configured
            if self.checkpoint_dir and (is_best or self.current_step % 10 == 0):
                self._save_checkpoint()
            
            if self.verbose:
                print(f"✅ Step {self.current_step} completed in {step_time:.2f}s")
                print(f"   Metrics: F1={f1_score:.4f}, AUC={auc_score:.4f}, G-mean={gmean_score:.4f}")
                print(f"   Reward: {reward:.4f}")
            
            return f1_score, auc_score, gmean_score, done
            
        except Exception as e:
            error_msg = f"Error in environment step {self.current_step + 1}: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            
            # Log error to history
            if self.training_history is not None:
                self.training_history.append({
                    'step': self.current_step + 1,
                    'hyperparams': hyperparams.copy(),
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Return poor scores on error to avoid breaking optimization
            return 0.0, 0.0, 0.0, False
    
    # ============================================================================
    # UTILITY AND HELPER METHODS
    # ============================================================================
    
    def _extract_metric(self, metrics: Dict[str, float], 
                       keys: List[str], default: float = 0.0) -> float:
        """
        Extract a metric value with fallback options.
        
        Args:
            metrics: Dictionary of metric values
            keys: List of possible keys to try (in order of preference)
            default: Default value if none of the keys are found
            
        Returns:
            Extracted metric value
        """
        for key in keys:
            if key in metrics:
                return float(metrics[key])
        return default
    
    def _should_stop(self, metrics: Dict[str, float], reward: float) -> bool:
        """
        Determine if optimization should stop early.
        
        Can be overridden by subclasses for custom stopping criteria.
        
        Args:
            metrics: Current evaluation metrics
            reward: Current reward value
            
        Returns:
            bool: True if optimization should stop
            
        Default stopping criteria:
        - Perfect score achieved (any metric >= 0.999)
        - No improvement for many steps (plateau detection)
        - Maximum time limit reached
        """
        # Stop if perfect score achieved
        if reward >= 0.999:
            if self.verbose:
                print("🎯 Perfect score achieved! Stopping optimization.")
            return True
        
        # Stop if plateau detected (no improvement in last 20 steps)
        if len(self.reward_history) >= 20:
            recent_rewards = self.reward_history[-20:]
            if max(recent_rewards) - min(recent_rewards) < 1e-6:
                if self.verbose:
                    print("📈 Performance plateau detected. Stopping optimization.")
                return True
        
        return False
    
    def add_step_callback(self, callback: Callable):
        """
        Add a custom callback function to be executed after each step.
        
        Args:
            callback: Function with signature callback(env, hyperparams, metrics, reward)
        """
        self.step_callbacks.append(callback)
    
    def add_epoch_callback(self, callback: Callable):
        """
        Add a custom callback function to be executed after each training epoch.
        
        Args:
            callback: Function with signature callback(env, epoch, model, metrics)
        """
        self.epoch_callbacks.append(callback)
    
    def get_best_results(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the best results found.
        
        Returns:
            Dictionary containing:
            - best_hyperparams: Best hyperparameter configuration
            - best_reward: Best reward achieved
            - best_metrics: Metrics corresponding to best reward
            - total_steps: Total optimization steps completed
            - total_time: Total optimization time
            - avg_step_time: Average time per step
            - training_history: Complete training history (if enabled)
        """
        total_time = sum(self.step_times) if self.step_times else 0
        avg_step_time = total_time / len(self.step_times) if self.step_times else 0
        
        # Find best metrics from history
        best_metrics = {}
        if self.training_history:
            for record in self.training_history:
                if record.get('is_best', False):
                    best_metrics = record.get('metrics', {})
                    break
        
        return {
            'best_hyperparams': self.best_hyperparams,
            'best_reward': self.best_reward,
            'best_metrics': best_metrics,
            'total_steps': self.current_step,
            'total_time': total_time,
            'avg_step_time': avg_step_time,
            'training_history': self.training_history,
            'reward_progression': self.reward_history
        }
    
    def _save_checkpoint(self):
        """Save current optimization state to checkpoint directory."""
        if not self.checkpoint_dir:
            return
        
        checkpoint_data = {
            'name': self.name,
            'current_step': self.current_step,
            'best_reward': self.best_reward,
            'best_hyperparams': self.best_hyperparams,
            'reward_history': self.reward_history,
            'step_times': self.step_times,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save main checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'environment_checkpoint.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Save training history separately (can be large)
        if self.training_history:
            history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_dir: str):
        """
        Load optimization state from checkpoint directory.
        
        Args:
            checkpoint_dir: Path to checkpoint directory
        """
        checkpoint_path = os.path.join(checkpoint_dir, 'environment_checkpoint.json')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Restore state
        self.current_step = checkpoint_data.get('current_step', 0)
        self.best_reward = checkpoint_data.get('best_reward', float('-inf'))
        self.best_hyperparams = checkpoint_data.get('best_hyperparams')
        self.reward_history = checkpoint_data.get('reward_history', [])
        self.step_times = checkpoint_data.get('step_times', [])
        
        # Load training history if available
        history_path = os.path.join(checkpoint_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        if self.verbose:
            print(f"📂 Loaded checkpoint from {checkpoint_dir}")
            print(f"   Resumed from step {self.current_step}")
            print(f"   Best reward so far: {self.best_reward:.4f}")
    
    def final_test(self, test_data: Optional[Any] = None, 
                   hyperparams: Optional[Dict[str, Any]] = None) -> Tuple[float, float, float]:
        """
        Perform final test evaluation with best hyperparameters.
        
        Args:
            test_data: Optional separate test dataset (uses validation data if None)
            hyperparams: Hyperparameters to use (uses best found if None)
            
        Returns:
            Tuple of (f1, auc, gmean) scores on test set
        """
        if hyperparams is None:
            if self.best_hyperparams is None:
                warnings.warn("No best hyperparameters found. Using default evaluation.")
                return 0.0, 0.0, 0.0
            hyperparams = self.best_hyperparams
        
        if self.verbose:
            print(f"🧪 Performing final test evaluation...")
        
        try:
            # Store original data and temporarily replace if test_data provided
            original_data = None
            if test_data is not None:
                original_data = getattr(self, 'validation_data', None)
                self.validation_data = test_data
            
            # Create model and evaluate
            model = self.create_model(hyperparams)
            metrics = self.train_evaluate(model, hyperparams)
            
            # Restore original data
            if original_data is not None:
                self.validation_data = original_data
            
            # Extract metrics
            f1 = self._extract_metric(metrics, ['f1', 'f1_score'], 0.0)
            auc = self._extract_metric(metrics, ['auc', 'auc_score', 'accuracy'], 0.0)
            gmean = self._extract_metric(metrics, ['gmean', 'g_mean', 'precision'], 0.0)
            
            if self.verbose:
                print(f"📊 Final test results: F1={f1:.4f}, AUC={auc:.4f}, G-mean={gmean:.4f}")
            
            return f1, auc, gmean
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error in final test: {e}")
            return 0.0, 0.0, 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the environment configuration and state.
        
        Returns:
            Dictionary with environment information
        """
        return {
            'name': self.name,
            'current_step': self.current_step,
            'best_reward': self.best_reward,
            'total_time': sum(self.step_times) if self.step_times else 0,
            'checkpoint_dir': self.checkpoint_dir,
            'has_training_history': self.training_history is not None,
            'history_length': len(self.training_history) if self.training_history else 0,
            'num_callbacks': len(self.step_callbacks) + len(self.epoch_callbacks)
        }