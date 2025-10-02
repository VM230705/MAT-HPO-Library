"""
Multi-Agent Transformer Hyperparameter Optimizer (MAT-HPO)

Sophisticated orchestration system for multi-agent reinforcement learning-based 
hyperparameter optimization. This module serves as the primary interface for users 
to conduct automated hyperparameter tuning using the SQDDPG algorithm.

Key Capabilities:
- Coordinated multi-agent learning with specialized hyperparameter responsibilities
- Advanced experience replay with prioritized sampling strategies
- Comprehensive monitoring, logging, and visualization of optimization progress
- Robust checkpoint/resume functionality for long-running optimization tasks
- Flexible configuration system supporting diverse optimization scenarios
- Early stopping mechanisms to prevent overfitting and reduce computation costs

The optimizer manages three specialized agents:
- Agent 0: Typically handles loss function parameters (class weights, regularization)
- Agent 1: Manages model architecture parameters (hidden sizes, layers)
- Agent 2: Controls training dynamics (learning rates, batch sizes, optimizers)

This modular design enables efficient exploration of complex hyperparameter spaces
while maintaining coordination between interdependent parameters.

Designed for enterprise-scale hyperparameter optimization with support for
distributed training, extensive logging, and production deployment considerations.
"""

import os
import json
import sys
import time
import torch
import numpy as np

from typing import Dict, Any, List, Optional, Tuple
from torch import optim
import torch.nn as nn

from .base_environment import BaseEnvironment
from .hyperparameter_space import HyperparameterSpace
from .agent import Actor, Critic, get_device
from .replay_buffer import TransReplayBuffer, Transition
from .sqddpg import SQDDPG
from ..utils.config import OptimizationConfig
from ..utils.logger import HPOLogger


class MAT_HPO_Optimizer:
    """
    Enterprise-Grade Multi-Agent Transformer Hyperparameter Optimizer
    
    This is the primary user-facing class for conducting sophisticated hyperparameter
    optimization using multi-agent reinforcement learning. It provides a comprehensive
    yet intuitive interface that abstracts away the complexity of coordinating multiple
    specialized agents while offering extensive customization and monitoring capabilities.
    
    Architecture Overview:
    The optimizer orchestrates three specialized agents working cooperatively:
    1. **Loss Agent (Agent 0)**: Optimizes loss function parameters, class weights,
       and regularization terms that directly affect the training objective
    2. **Architecture Agent (Agent 1)**: Manages model structure parameters like
       hidden layer sizes, number of layers, and architectural choices
    3. **Training Agent (Agent 2)**: Controls training dynamics including learning
       rates, batch sizes, optimizer selection, and scheduling parameters
    
    Key Features:
    - **Intelligent Exploration**: Uses Shapley value-based credit assignment to
      fairly distribute rewards among agents, leading to more efficient exploration
    - **Adaptive Learning**: Dynamic learning rate adjustment and prioritized
      experience replay ensure stable convergence across diverse problem domains
    - **Production Ready**: Comprehensive logging, checkpointing, and monitoring
      capabilities designed for enterprise-scale deployments
    - **Flexible Integration**: Easy integration with existing ML pipelines through
      the BaseEnvironment abstraction
    - **Resource Efficient**: Built-in early stopping and intelligent sampling
      strategies minimize computational overhead
    
    Typical Usage Pattern:
    ```python
    # Define your optimization environment
    class MyEnvironment(BaseEnvironment):
        def step(self, hyperparams):
            # Your training/evaluation logic here
            return f1_score, auc_score, gmean_score, done
    
    # Configure hyperparameter space
    hyp_space = HyperparameterSpace()
    hyp_space.add_continuous('learning_rate', 1e-5, 1e-2, agent=2)
    hyp_space.add_discrete('batch_size', [16, 32, 64, 128], agent=2)
    
    # Run optimization
    optimizer = MAT_HPO_Optimizer(environment, hyp_space, config)
    results = optimizer.optimize()
    ```
    """
    
    def __init__(self,
                 environment: BaseEnvironment,
                 hyperparameter_space: HyperparameterSpace,
                 config: OptimizationConfig,
                 output_dir: str = "./mat_hpo_results"):
        """
        Initialize the MAT-HPO optimizer with comprehensive setup and validation.
        
        This constructor performs extensive initialization including:
        1. Environment and hyperparameter space validation
        2. Multi-agent system setup with specialized roles
        3. Experience replay buffer configuration
        4. Optimizer and learning rate scheduler initialization
        5. Logging and monitoring system setup
        6. Checkpoint and state management preparation
        
        Args:
            environment: User-defined environment implementing BaseEnvironment.
                        Must provide step() method that evaluates hyperparameters
                        and returns performance metrics (f1, auc, gmean, done).
            hyperparameter_space: Comprehensive definition of the search space
                                including parameter types, bounds, and agent assignments.
                                Must define parameters for all three agents.
            config: OptimizationConfig object containing all optimization settings
                   including learning rates, batch sizes, update frequencies,
                   early stopping criteria, and device preferences.
            output_dir: Directory path for saving optimization results, model
                       checkpoints, logs, and visualizations. Created automatically
                       if it doesn't exist. Default: "./mat_hpo_results"
        
        Raises:
            ValueError: If environment doesn't implement required methods
            ValueError: If hyperparameter_space is empty or invalid
            ValueError: If config contains invalid parameters
            OSError: If output_dir cannot be created or accessed
        """
        self.environment = environment
        self.hyperparameter_space = hyperparameter_space
        self.config = config
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup device
        self.device = config.device
        
        # Initialize tracking variables
        self.current_step = 0
        self.best_reward = float('-inf')
        self.best_hyperparams = None
        self.training_history = []

        # Support flexible number of metrics (backward compatible with f1/auc/gmean)
        self.best_f1 = 0.0
        self.best_auc = 0.0
        self.best_gmean = 0.0
        self.best_metrics = {}  # Flexible storage for any number of metrics

        # Extract metric names from environment if available
        metric_names = getattr(environment, 'metric_names_mapping', None)
        custom_metrics = getattr(environment, 'custom_metrics', None)

        # Initialize logger with custom metrics support
        self.logger = HPOLogger(
            output_dir,
            verbose=config.verbose,
            metric_names=metric_names,
            custom_metrics=custom_metrics
        )

        # Store metric names for later use
        self.metric_names = metric_names
        
        # Move hyperparameter space to device
        self.hyperparameter_space.to_device(self.device)
        
        # Initialize multi-agent system
        self._initialize_agents()
        
        # Paths for saving
        self.temp_model_path = os.path.join(output_dir, 'temp_model.pt')
        self.best_model_path = os.path.join(output_dir, 'best_model.pt')
        
        self.logger.info("MAT-HPO Optimizer initialized successfully")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(hyperparameter_space.get_summary())
    
    def _initialize_agents(self):
        """
        Initialize the complete multi-agent reinforcement learning system.
        
        This method sets up the core SQDDPG (Shapley Q-Value Deep Deterministic Policy
        Gradient) algorithm with three specialized agents, their corresponding neural
        networks, optimizers, and the experience replay infrastructure.
        
        **System Architecture**:
        1. **Agent Initialization**: Creates three Actor networks (one per agent) with
           dimensions matching their assigned hyperparameters
        2. **Critic Networks**: Initializes three Critic networks for value function
           estimation, each taking joint observations and actions as input
        3. **Experience Replay**: Sets up prioritized replay buffer for stable learning
        4. **Optimizers**: Configures Adam optimizers with separate learning rates
           for actor (policy) and critic (value) networks
        
        **Key Design Decisions**:
        - Separate optimizers allow different learning rates for exploration vs exploitation
        - Prioritized replay buffer improves sample efficiency by focusing on informative transitions
        - Agent-specific network dimensions enable specialized parameter handling
        - Device placement ensures efficient GPU utilization when available
        """
        # Get dimensions for each agent
        hyp_num0, hyp_num1, hyp_num2 = self.hyperparameter_space.agent_dims
        
        # Initialize SQDDPG with agent dimensions
        self.sqddpg = SQDDPG(hyp_num0, hyp_num1, hyp_num2, device=self.device)
        
        # Initialize replay buffer
        self.replay_buffer = TransReplayBuffer(self.config.replay_buffer_size)
        
        # Initialize optimizers
        self.action_optimizers = [
            optim.Adam(self.sqddpg.actor0.parameters(), lr=self.config.policy_learning_rate),
            optim.Adam(self.sqddpg.actor1.parameters(), lr=self.config.policy_learning_rate),
            optim.Adam(self.sqddpg.actor2.parameters(), lr=self.config.policy_learning_rate)
        ]
        
        self.value_optimizers = [
            optim.Adam(self.sqddpg.critic0.parameters(), lr=self.config.value_learning_rate),
            optim.Adam(self.sqddpg.critic1.parameters(), lr=self.config.value_learning_rate),
            optim.Adam(self.sqddpg.critic2.parameters(), lr=self.config.value_learning_rate)
        ]
    
    def optimize(self) -> Dict[str, Any]:
        """
        Execute the complete multi-agent hyperparameter optimization process.
        
        This is the main method that orchestrates the entire optimization workflow:
        
        **Optimization Loop**:
        1. **Action Generation**: Each agent generates actions based on current state
        2. **Hyperparameter Translation**: Actions are converted to actual hyperparameter values
        3. **Environment Evaluation**: Hyperparameters are evaluated in the user's environment
        4. **Reward Computation**: Performance metrics are converted to reward signals
        5. **Experience Storage**: Transitions are stored in prioritized replay buffer
        6. **Agent Updates**: Periodic updates to actor-critic networks using Shapley values
        7. **Progress Tracking**: Comprehensive logging and monitoring of optimization progress
        
        **Advanced Features**:
        - **Adaptive Exploration**: Exploration strategy adapts based on performance history
        - **Early Stopping**: Automatic termination when convergence is detected
        - **Best Model Tracking**: Continuous monitoring and saving of best-performing configurations
        - **Real-time Visualization**: Live progress updates and hyperparameter selection display
        - **Robust Error Handling**: Graceful handling of evaluation failures and edge cases
        
        **Convergence Criteria**:
        The optimization terminates when any of the following conditions are met:
        - Maximum number of steps reached (config.max_steps)
        - Early stopping criteria satisfied (no improvement for specified patience)
        - Environment signals completion (done=True from environment.step())
        - User interruption (Ctrl+C) with graceful cleanup
        
        Returns:
            Comprehensive results dictionary containing:
            - 'best_hyperparameters': Optimal hyperparameter configuration found
            - 'best_performance': Performance metrics (f1, auc, gmean) of best config
            - 'optimization_stats': Detailed statistics about the optimization process
            - 'training_history': Complete log of all evaluated configurations
            - 'agent_statistics': Individual performance metrics for each agent
            - 'convergence_info': Analysis of convergence behavior and stopping criteria
            
        Raises:
            RuntimeError: If optimization fails due to environment or agent errors
            KeyboardInterrupt: If user interrupts optimization (handled gracefully)
            ValueError: If invalid configurations are encountered during optimization
        """
        self.logger.info(f"🚀 Starting MAT-HPO optimization for {self.config.max_steps} steps")
        
        # Get parameter count in a compatible way
        if hasattr(self.hyperparameter_space, 'parameters'):
            # New dynamic initialization
            param_count = len(self.hyperparameter_space.parameters)
        else:
            # Legacy initialization
            param_count = sum(self.hyperparameter_space.agent_dims)
        
        self.logger.info(f"📊 Search space: {param_count} parameters across 3 agents")
        
        # Display hyperparameter search space at startup (if available)
        if hasattr(self.hyperparameter_space, 'parameters'):
            print(f"\n🔍 超參數搜尋空間:")
            print("=" * 50)
            for param_name, param_info in self.hyperparameter_space.parameters.items():
                agent_id = param_info.get('agent', 0)
                param_type = param_info.get('type', 'continuous')
                if param_type == 'continuous':
                    min_val, max_val = param_info['bounds']
                    print(f"Agent {agent_id}: {param_name} ∈ [{min_val:.2e}, {max_val:.2e}]")
                elif param_type == 'discrete':
                    choices = param_info['choices']
                    print(f"Agent {agent_id}: {param_name} ∈ {choices}")
            print("=" * 50)
        start_time = time.time()
        
        # Initialize environment
        state = self.environment.reset().to(self.device)
        
        for step in range(self.config.max_steps):
            self.current_step = step
            step_start_time = time.time()
            
            # Generate actions using SQDDPG
            actions = self.sqddpg.policy(state)
            selected_actions = self.sqddpg.select_action(actions)
            
            # Extract hyperparameters from actions
            hyperparams = self._extract_hyperparameters(selected_actions)
            
            # Display selected hyperparameters for this step
            self._display_step_hyperparameters(step, hyperparams)
            
            # Evaluate hyperparameters in environment
            f1, auc, gmean, done = self.environment.step(hyperparams)
            
            step_time = time.time() - step_start_time
            
            # Log step results
            self.logger.log_step(step, f1, auc, gmean, step_time, hyperparams)
            
            # Track best results
            reward = gmean * 100  # Convert to reward scale
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_hyperparams = hyperparams.copy()
                self.best_f1, self.best_auc, self.best_gmean = f1, auc, gmean

                # Store metrics flexibly (support any number of metrics from hyperparams)
                self.best_metrics = {}
                for key, value in hyperparams.items():
                    if key.startswith('original_') or key in ['mase', 'train_loss', 'val_loss', 'overfitting_ratio', 'mse', 'mape', 'msmape', 'smape', 'mae', 'rmse']:
                        self.best_metrics[key] = value

                self._save_best_model()  # 這個方法已經包含了保存best_hyperparams.json的邏輯
                self._save_rl_model_input(state)  # Save current state as RL_model_input.pt

                # Log with custom metric names if available
                if self.metric_names:
                    metric1_name = self.metric_names.get('f1', 'F1')
                    metric2_name = self.metric_names.get('auc', 'AUC')
                    metric3_name = self.metric_names.get('gmean', 'G-mean')
                    self.logger.info(f"🎯 New best model at step {step}! "
                                   f"{metric1_name}={f1:.4f}, {metric2_name}={auc:.4f}, {metric3_name}={gmean:.4f}")
                else:
                    self.logger.info(f"🎯 New best model at step {step}! "
                                   f"F1={f1:.4f}, AUC={auc:.4f}, G-mean={gmean:.4f}")
            
            # Update replay buffer
            reward_array = np.array([reward] * len(self.action_optimizers))
            transition = Transition(
                state.cpu().numpy(),
                selected_actions.detach().cpu().numpy(),
                reward_array
            )
            self.replay_buffer.add_experience(transition)
            
            # Update agents if enough samples - with enhanced error handling
            if (len(self.replay_buffer) >= self.config.batch_size and 
                step % self.config.behaviour_update_freq == 0):
                try:
                    self._update_agents()
                except Exception as e:
                    print(f"⚠️ Agent update failed at step {step}: {e}")
                    # Continue optimization even if agent update fails
                    pass
            
            # Early stopping check
            if done or self._check_early_stopping():
                self.logger.info(f"Early stopping at step {step}")
                break
            
            # Update state for next iteration
            state = self.environment.reset().to(self.device)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Save final results
        results = self._save_final_results(total_time)
        
        self.logger.info(f"Optimization completed in {total_time:.2f} seconds")
        self.logger.info(f"Best hyperparameters: {self.best_hyperparams}")
        self.logger.info(f"Best performance: F1={self.best_f1:.4f}, "
                        f"AUC={self.best_auc:.4f}, G-mean={self.best_gmean:.4f}")
        
        return results
    
    def _extract_hyperparameters(self, actions: torch.Tensor) -> Dict[str, Any]:
        """
        Translate raw agent actions into concrete hyperparameter values.
        
        This critical method bridges the gap between the continuous action space
        of the reinforcement learning agents and the discrete/bounded hyperparameter
        space required by the optimization environment.
        
        **Translation Process**:
        1. **Action Splitting**: Separates the joint action tensor into agent-specific actions
        2. **Dimension Alignment**: Extracts only the relevant dimensions for each agent
           based on their assigned hyperparameter counts
        3. **Value Mapping**: Uses the HyperparameterSpace to convert normalized actions
           (-1 to 1) into actual hyperparameter values within specified bounds
        4. **Type Conversion**: Ensures proper data types (int, float, categorical)
           for each hyperparameter according to its definition
        
        **Agent Responsibilities**:
        - Agent 0: Converts actions to class weights, regularization parameters
        - Agent 1: Maps actions to architecture choices (layer sizes, depths)
        - Agent 2: Translates actions to training parameters (lr, batch_size, etc.)
        
        Args:
            actions: Raw action tensor from SQDDPG with shape [batch_size=1, n_agents=3, max_dim]
                    where max_dim is the maximum hyperparameter count across all agents
            
        Returns:
            Dictionary mapping hyperparameter names to their concrete values,
            ready for use in the optimization environment. All values are properly
            typed and within their specified bounds/constraints.
            
        Example:
            Input actions: tensor([[[0.2, -0.5], [0.8, 0.1], [-0.3, 0.7]]])
            Output: {'learning_rate': 0.001, 'batch_size': 64, 'hidden_size': 128, ...}
        """
        # Split actions for each agent
        action_list = []
        for i in range(3):
            agent_dim = self.hyperparameter_space.agent_dims[i]
            agent_actions = actions[0, i, :agent_dim]
            action_list.append(agent_actions)
        
        # Translate actions to hyperparameter values
        hyperparams = self.hyperparameter_space.translate_actions(action_list)
        
        return hyperparams
    
    def _update_agents(self):
        """
        Coordinate the training update process for all agents using experience replay.
        
        This method orchestrates the core learning procedure of the SQDDPG algorithm,
        implementing a sophisticated update strategy that balances exploration and
        exploitation while maintaining stable learning across all three agents.
        
        **Update Strategy**:
        1. **Critic Updates**: Multiple updates to value functions using prioritized
           experience replay to improve value estimation accuracy
        2. **Actor Updates**: Single policy update per cycle to prevent instability
           and maintain exploration capabilities
        3. **Gradient Management**: Careful gradient flow control with clipping
           and retain_graph management for multi-agent backpropagation
        
        **Learning Dynamics**:
        - Critics are updated more frequently (config.critic_update_times) to provide
          stable value estimates for policy gradient computation
        - Actors are updated less frequently to prevent rapid policy changes that
          could destabilize the learning process
        - Experience sampling prioritizes high-reward transitions for faster convergence
        
        The method only triggers updates when sufficient experience has been collected
        (batch_size threshold) and respects the configured update frequency to balance
        learning progress with computational efficiency.
        """
        # Update value functions
        for _ in range(self.config.critic_update_times):
            self._update_critics()
        
        # Update policy
        self._update_actors()
    
    def _update_critics(self):
        """Update critic networks with enhanced error handling"""
        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        try:
            batch = self.replay_buffer.get_batch(self.config.batch_size)
            batch_transitions = self.sqddpg.Transition(*zip(*batch))
            
            # Disable anomaly detection to avoid performance issues
            torch.autograd.set_detect_anomaly(False)
            _, value_losses, _ = self.sqddpg.get_loss(batch_transitions)
            
            # Validate losses
            valid_losses = []
            for i, loss in enumerate(value_losses):
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Invalid loss value in critic {i}: {loss.item()}")
                else:
                    valid_losses.append(loss)
            if not valid_losses:
                return
            
            # Combine losses to avoid retain_graph/backward-on-stale-graph issues
            combined_loss = sum(valid_losses)
            
            # Zero all critic grads
            for opt in self.value_optimizers:
                opt.zero_grad()
            
            # Backward once through a single graph
            combined_loss.backward()
            
            # Gradient clipping (safe, non-inplace)
            for i, opt in enumerate(self.value_optimizers):
                params = [p for group in opt.param_groups for p in group['params'] if p.grad is not None]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, self.config.gradient_clip)
            
            # Step all optimizers after gradients are computed
            for opt in self.value_optimizers:
                opt.step()
            
        except Exception as e:
            print(f"⚠️ Critical error in _update_critics: {e}")
            # Clear all gradients to prevent accumulation
            for optimizer in self.value_optimizers:
                optimizer.zero_grad()
    
    def _update_actors(self):
        """Update actor networks with enhanced error handling"""
        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        try:
            batch = self.replay_buffer.get_batch(self.config.batch_size)
            batch_transitions = self.sqddpg.Transition(*zip(*batch))
            
            # Disable anomaly detection during training step for speed
            torch.autograd.set_detect_anomaly(False)
            action_losses, _, _ = self.sqddpg.get_loss(batch_transitions)
            
            # Validate losses and combine to avoid retain_graph complications
            valid_losses = []
            for i, loss in enumerate(action_losses):
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Invalid loss value in actor {i}: {loss.item()}")
                else:
                    valid_losses.append(loss)
            if not valid_losses:
                return
            combined_loss = sum(valid_losses)
            
            # Zero all actor grads
            for opt in self.action_optimizers:
                opt.zero_grad()
            
            # Freeze critics to prevent gradient updates to critic parameters during actor step
            critics = [self.sqddpg.critic0, self.sqddpg.critic1, self.sqddpg.critic2]
            try:
                for c in critics:
                    for p in c.parameters():
                        p.requires_grad_(False)
                
                # Single backward pass through shared graph
                combined_loss.backward()
            finally:
                # Unfreeze critics regardless of success
                for c in critics:
                    for p in c.parameters():
                        p.requires_grad_(True)
            
            # Gradient clipping (safe, non-inplace)
            for opt in self.action_optimizers:
                params = [p for group in opt.param_groups for p in group['params'] if p.grad is not None]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, self.config.gradient_clip)
            
            # Step all actor optimizers
            for opt in self.action_optimizers:
                opt.step()
        
        except Exception as e:
            print(f"⚠️ Critical error in _update_actors: {e}")
            # Clear all gradients to prevent accumulation
            for optimizer in self.action_optimizers:
                optimizer.zero_grad()
    
    def _save_best_model(self):
        """
        Persist the current best-performing model state and associated hyperparameters.
        
        This method implements comprehensive checkpointing functionality that saves
        both the neural network parameters and the corresponding hyperparameter
        configuration that achieved the best performance so far.
        
        **Saved Artifacts**:
        1. **Hyperparameter Configuration**: Complete set of hyperparameters and
           their performance metrics saved in JSON format for easy inspection
        2. **Agent Models**: Individual state dictionaries for all three actor networks
           allowing for precise model reconstruction and analysis
        3. **Performance Metadata**: Detailed performance metrics, optimization step,
           and timestamp information for tracking progress over time
        
        **Use Cases**:
        - **Resume Optimization**: Load saved models to continue optimization from best state
        - **Model Analysis**: Examine which hyperparameters led to best performance
        - **Production Deployment**: Use best model configuration in production systems
        - **Reproducibility**: Ensure exact reproduction of best results
        
        The saved files follow a consistent naming convention and include comprehensive
        metadata to support various downstream applications and analysis workflows.
        """
        # Save hyperparameters
        best_hyp_path = os.path.join(self.output_dir, 'best_hyperparams.json')
        
        # 使用自定義指標名稱（如果可用），否則使用默認名稱
        if hasattr(self, 'metric_names') and self.metric_names:
            metric_key1 = self.metric_names.get('f1', 'f1').lower()
            metric_key2 = self.metric_names.get('auc', 'auc').lower()
            metric_key3 = self.metric_names.get('gmean', 'gmean').lower()
        else:
            metric_key1, metric_key2, metric_key3 = 'f1', 'auc', 'gmean'
        
        # Build performance dict with flexible metric support
        performance = {}

        # Add all metrics from best_metrics (most flexible approach)
        for key, value in self.best_metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                # Remove 'original_' prefix for cleaner output
                clean_key = key.replace('original_', '') if key.startswith('original_') else key
                performance[clean_key] = float(value)

        # Backward compatibility: add f1/auc/gmean if using custom names
        if self.metric_names:
            # Map the three primary metrics
            performance[metric_key1] = float(self.best_f1)
            performance[metric_key2] = float(self.best_auc)
            performance[metric_key3] = float(self.best_gmean)
        else:
            # Default metric names
            performance['f1'] = float(self.best_f1)
            performance['auc'] = float(self.best_auc)
            performance['gmean'] = float(self.best_gmean)

        # Always include reward
        performance['reward'] = float(self.best_reward)

        with open(best_hyp_path, 'w') as f:
            json.dump({
                'hyperparameters': self.best_hyperparams,
                'performance': performance,
                'step': self.current_step
            }, f, indent=2)
        
        # Note: Removed best_actor*.pt files as they are redundant with RL_model*.pt
        
        # Save RL models in original MAT_HPO format for compatibility
        torch.save(self.sqddpg.actor0, os.path.join(self.output_dir, 'RL_model0.pt'))
        torch.save(self.sqddpg.actor1, os.path.join(self.output_dir, 'RL_model1.pt'))
        torch.save(self.sqddpg.actor2, os.path.join(self.output_dir, 'RL_model2.pt'))
        
        # Save hyperparameters in numpy format (original MAT_HPO compatibility)
        import numpy as np
        if self.best_hyperparams:
            # Convert hyperparameters to list format compatible with original system
            hyp_list = self._convert_hyperparams_to_list(self.best_hyperparams)
            np.save(os.path.join(self.output_dir, 'CNNLSTM_model_hyp.npy'), np.array(hyp_list))
    
    def _convert_hyperparams_to_list(self, hyperparams: Dict[str, Any]) -> List[float]:
        """
        Convert hyperparameters dictionary to list format compatible with original MAT_HPO system.
        
        Original MAT_HPO expects hyperparameters in this order:
        - Class weights (varies by dataset)
        - Architecture params: lstm_dim, conv_out1, conv_out2, conv_out3, fc_out2, fc_out3, att_dim, dilation, rp_group
        - Training params: batch_size, learning_rate
        """
        hyp_list = []
        
        # Add class weights (Agent 0 parameters)
        class_weight_keys = [k for k in hyperparams.keys() if k.startswith('class_weight_')]
        class_weight_keys.sort()  # Ensure consistent ordering
        for key in class_weight_keys:
            hyp_list.append(float(hyperparams[key]))
        
        # Add architecture parameters (Agent 1 parameters)  
        arch_params = ['lstm_dim', 'conv_out1', 'conv_out2', 'conv_out3', 'fc_out2', 'fc_out3', 'att_dim', 'dilation', 'rp_group']
        for param in arch_params:
            if param in hyperparams:
                hyp_list.append(float(hyperparams[param]))
            else:
                # Use default values if parameter not found
                default_values = {
                    'lstm_dim': 48, 'conv_out1': 48, 'conv_out2': 48, 'conv_out3': 48,
                    'fc_out2': 48, 'fc_out3': 48, 'att_dim': 48, 'dilation': 3, 'rp_group': 5
                }
                hyp_list.append(float(default_values.get(param, 48)))
        
        # Add training parameters (Agent 2 parameters)
        training_params = ['batch_size', 'learning_rate']
        for param in training_params:
            if param in hyperparams:
                hyp_list.append(float(hyperparams[param]))
            else:
                # Use default values
                default_values = {'batch_size': 32, 'learning_rate': 1e-3}
                hyp_list.append(float(default_values.get(param, 1e-3)))
        
        return hyp_list
    
    def _save_rl_model_input(self, state: torch.Tensor):
        """
        Save the current state as RL_model_input.pt for compatibility with original MAT_HPO.
        """
        torch.save(state.cpu(), os.path.join(self.output_dir, 'RL_model_input.pt'))
    
    def _check_early_stopping(self) -> bool:
        """
        Evaluate whether early stopping criteria have been satisfied to prevent overfitting.
        
        This method implements intelligent convergence detection to automatically terminate
        optimization when continued training is unlikely to yield significant improvements.
        This prevents computational waste and reduces the risk of overfitting to the
        hyperparameter optimization objective.
        
        **Early Stopping Logic**:
        - Monitors recent performance history over a configurable patience window
        - Detects convergence by measuring variance in recent reward values
        - Uses configurable threshold to balance between premature stopping and
          unnecessary computation continuation
        
        **Benefits**:
        1. **Resource Efficiency**: Prevents unnecessary computation when convergence is achieved
        2. **Overfitting Prevention**: Stops before the model starts overfitting to the HP space
        3. **Automatic Termination**: Reduces need for manual monitoring of long optimizations
        4. **Adaptive Behavior**: Threshold-based approach adapts to different problem scales
        
        Returns:
            bool: True if early stopping criteria are met (optimization should terminate),
                 False if optimization should continue.
                 
        Note:
            Early stopping is disabled if patience <= 0 in configuration, allowing
            for full exploration of the specified number of optimization steps.
        """
        if self.config.early_stop_patience <= 0:
            return False
        
        # Simple early stopping: no improvement for patience steps
        if len(self.training_history) < self.config.early_stop_patience:
            return False
        
        recent_rewards = [h['reward'] for h in self.training_history[-self.config.early_stop_patience:]]
        return max(recent_rewards) - min(recent_rewards) < self.config.early_stop_threshold
    
    def _save_final_results(self, total_time: float) -> Dict[str, Any]:
        """
        Generate and persist comprehensive final optimization results and analytics.
        
        This method creates a detailed summary of the entire optimization process,
        including performance metrics, timing statistics, configuration details,
        and analysis insights. The results are both returned for immediate use
        and saved to disk for future reference and analysis.
        
        **Result Components**:
        1. **Best Configuration**: Optimal hyperparameters and their performance
        2. **Optimization Statistics**: Timing, convergence, and efficiency metrics
        3. **Configuration Archive**: Complete record of all optimization settings
        4. **Environment Context**: Information about the optimization environment
        5. **Search Space Summary**: Documentation of explored hyperparameter space
        
        **Analytics and Insights**:
        - Performance tracking over time
        - Convergence behavior analysis
        - Resource utilization statistics
        - Search space coverage metrics
        
        Args:
            total_time: Total wall-clock time spent on optimization in seconds
            
        Returns:
            Dict[str, Any]: Comprehensive results dictionary suitable for
                          analysis, reporting, and integration with downstream
                          systems. Also saved to 'optimization_results.json'.
                          
        **File Output**:
        Results are automatically saved to the output directory as a JSON file
        with proper formatting for human readability and programmatic access.
        """
        # 使用自定義指標名稱（如果可用），否則使用默認名稱
        if hasattr(self, 'metric_names') and self.metric_names:
            metric_key1 = self.metric_names.get('f1', 'f1').lower()
            metric_key2 = self.metric_names.get('auc', 'auc').lower()
            metric_key3 = self.metric_names.get('gmean', 'gmean').lower()
        else:
            metric_key1, metric_key2, metric_key3 = 'f1', 'auc', 'gmean'
        
        results = {
            'best_hyperparameters': self.best_hyperparams,
            'best_performance': {
                metric_key1: float(self.best_f1),
                metric_key2: float(self.best_auc), 
                metric_key3: float(self.best_gmean),
                'reward': float(self.best_reward)
            },
            'optimization_stats': {
                'total_steps': self.current_step + 1,
                'total_time': total_time,
                'avg_time_per_step': total_time / (self.current_step + 1)
            },
            'config': self.config.to_dict(),
            'environment_name': self.environment.name,
            'hyperparameter_space': self.hyperparameter_space.get_summary()
        }
        
        # Save to file
        results_path = os.path.join(self.output_dir, 'optimization_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def get_best_hyperparameters(self) -> Dict[str, Any]:
        """
        Retrieve the optimal hyperparameter configuration discovered during optimization.
        
        This method provides access to the best hyperparameter set found throughout
        the optimization process, along with comprehensive metadata about when and
        how this configuration was discovered.
        
        **Returned Information**:
        The returned dictionary contains the complete hyperparameter configuration
        that achieved the highest reward (typically based on G-mean performance).
        All hyperparameters are returned in their original format and data types
        as specified in the HyperparameterSpace definition.
        
        **Safety and Reliability**:
        - Returns a deep copy to prevent accidental modification of internal state
        - Handles the case where no optimization has been run (returns empty dict)
        - Maintains consistency with the original hyperparameter space definitions
        
        Returns:
            Dict[str, Any]: Complete hyperparameter configuration with parameter names
                           as keys and their optimal values as values. Returns empty
                           dictionary if no optimization has been performed yet.
                           
        Example:
            ```python
            optimizer = MAT_HPO_Optimizer(...)
            results = optimizer.optimize()
            best_params = optimizer.get_best_hyperparameters()
            
            # Use best parameters for final model training
            model = create_model(**best_params)
            ```
        """
        return self.best_hyperparams.copy() if self.best_hyperparams else {}
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Restore optimizer state from a previously saved checkpoint for resuming optimization.
        
        This method enables seamless continuation of long-running optimization processes
        by restoring all critical state information including model parameters, best
        performance metrics, and optimization progress.
        
        **Restoration Process**:
        1. **Performance State**: Loads best hyperparameters and their associated
           performance metrics (F1, AUC, G-mean scores)
        2. **Model Parameters**: Restores trained weights for all three actor networks
           to continue learning from the saved state
        3. **Optimization Context**: Recovers optimization step count and progress
           information for accurate continuation
        
        **Robust Loading**:
        - Gracefully handles missing files by skipping unavailable components
        - Validates checkpoint compatibility with current configuration
        - Provides detailed logging of loaded components for transparency
        - Uses proper device mapping to ensure compatibility across different hardware
        
        **Use Cases**:
        - **Long Optimizations**: Resume multi-day optimization runs after interruption
        - **Incremental Tuning**: Extend optimization with additional steps or new constraints
        - **Distributed Computing**: Transfer optimization state between different machines
        - **Experimental Iteration**: Compare different continuation strategies from same baseline
        
        Args:
            checkpoint_path: Directory path containing the checkpoint files.
                           Should contain 'best_hyperparams.json' and model files
                           ('best_actor0.pt', 'best_actor1.pt', 'best_actor2.pt').
        
        Raises:
            FileNotFoundError: If checkpoint_path doesn't exist
            ValueError: If checkpoint files are corrupted or incompatible
            RuntimeError: If model loading fails due to architecture mismatch
        """
        # Load hyperparameters
        hyp_path = os.path.join(checkpoint_path, 'best_hyperparams.json')
        if os.path.exists(hyp_path):
            with open(hyp_path, 'r') as f:
                data = json.load(f)
                self.best_hyperparams = data['hyperparameters']
                perf = data['performance']

                # Load metrics flexibly
                self.best_f1 = perf.get('f1', perf.get('smape', 0.0))
                self.best_auc = perf.get('auc', perf.get('mae', 0.0))
                self.best_gmean = perf.get('gmean', perf.get('rmse', 0.0))
                self.best_reward = perf.get('reward', 0.0)

                # Load all available metrics into best_metrics
                self.best_metrics = {k: v for k, v in perf.items() if k != 'reward'}
        
        # Load models if they exist
        model_paths = [
            ('best_actor0.pt', self.sqddpg.actor0),
            ('best_actor1.pt', self.sqddpg.actor1), 
            ('best_actor2.pt', self.sqddpg.actor2)
        ]
        
        for filename, model in model_paths:
            path = os.path.join(checkpoint_path, filename)
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=self.device))
                self.logger.info(f"Loaded {filename}")
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def _display_step_hyperparameters(self, step: int, hyperparams: Dict[str, Any]) -> None:
        """
        Display the hyperparameters selected by each agent for the current optimization step.
        
        This method provides real-time visibility into the decision-making process of each
        specialized agent, helping users understand the exploration strategy and parameter
        interactions. The display is organized by agent responsibility for clarity.
        
        Args:
            step: Current optimization step number (0-indexed)
            hyperparams: Dictionary of hyperparameter names and their selected values
                        as determined by the three agents' actions
        
        Display Format:
        - Agent 0 (Loss Parameters): Class weights, regularization terms, loss function params
        - Agent 1 (Architecture): Hidden layer sizes, number of layers, model structure
        - Agent 2 (Training): Learning rates, batch sizes, optimizer settings, schedules
        
        Note:
            This method uses Chinese labels (Agent 0: 📊, Agent 1: 🏗️, Agent 2: ⚙️) for
            better visualization and can be easily localized for different languages.
        """
        print(f"\nStep {step + 1} - Selected hyperparameters:")
        print("-" * 45)
        
        # Agent 0: 類別權重 (顯示前3個和總結)
        class_weight_params = [k for k in hyperparams.keys() if k.startswith('class_weight_')]
        if class_weight_params:
            class_weights = [hyperparams[k] for k in sorted(class_weight_params)]
            if len(class_weights) <= 3:
                weight_str = ', '.join([f"{w:.2f}" for w in class_weights])
            else:
                weight_str = f"{class_weights[0]:.2f}, {class_weights[1]:.2f}, {class_weights[2]:.2f}...({len(class_weights)} total)"
            print(f"Agent 0 (Class Weights): [{weight_str}]")
        
        # Agent 1: 架構參數
        agent1_params = []
        if 'hidden_size' in hyperparams:
            agent1_params.append(f"hidden_size = {hyperparams['hidden_size']}")
        if 'loss_function' in hyperparams:
            agent1_params.append(f"loss_function = {hyperparams['loss_function']}")
        
        if agent1_params:
            print(f"Agent 1 (Architecture): {', '.join(agent1_params)}")
        
        # Agent 2: 訓練參數
        training_params = []
        if 'batch_size' in hyperparams:
            training_params.append(f"batch_size={hyperparams['batch_size']}")
        if 'learning_rate' in hyperparams:
            training_params.append(f"lr={hyperparams['learning_rate']:.2e}")
        
        if training_params:
            print(f"Agent 2 (Training): {', '.join(training_params)}")
        
        print("-" * 45)