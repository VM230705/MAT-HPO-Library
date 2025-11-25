"""
LLM Enhanced MAT-HPO Optimizer
Integrates LLM guidance with multi-agent optimization
"""

import numpy as np
import torch
import time
from typing import Dict, Any, List, Optional, Tuple
from .multi_agent_optimizer import MAT_HPO_Optimizer
from .evaluation_criteria import ModelSaveCriteria, FlexibleEvaluator, OptimizationTarget, create_spnv2_criteria
from ..llm import EnhancedLLMHyperparameterMixer, OllamaLLMClient
from ..utils.config import OptimizationConfig
from .replay_buffer import Transition


class LLMEnhancedOptimizationConfig(OptimizationConfig):
    """Enhanced optimization config with LLM parameters"""

    def __init__(self, **kwargs):
        # Extract LLM-specific parameters before calling super()
        llm_kwargs = {}
        llm_params = [
            'enable_llm', 'llm_model', 'llm_base_url', 'mixing_strategy', 'alpha',
            'slope_threshold', 'llm_cooldown_episodes', 'min_episodes_before_llm',
            'custom_prompt_template', 'dataset_name', 'task_description',
            'llm_search_space_mode', 'llm_parameter_bounds', 'llm_parameter_types',
            'dataset_info_csv_path'  # 新增参数
        ]

        for param in llm_params:
            if param in kwargs:
                llm_kwargs[param] = kwargs.pop(param)

        # Call parent constructor with remaining kwargs
        super().__init__(**kwargs)

        # LLM configuration
        self.enable_llm = llm_kwargs.get('enable_llm', False)  # Default to False for backward compatibility
        self.llm_model = llm_kwargs.get('llm_model', 'llama3.2:3b')
        self.llm_base_url = llm_kwargs.get('llm_base_url', 'http://localhost:11434')
        self.dataset_info_csv_path = llm_kwargs.get('dataset_info_csv_path', './Datasets_info.csv')  # 新增参数

        # Mixing strategy configuration
        self.mixing_strategy = llm_kwargs.get('mixing_strategy', 'fixed_alpha')
        self.alpha = llm_kwargs.get('alpha', 0.3)
        self.slope_threshold = llm_kwargs.get('slope_threshold', 0.01)

        # LLM call frequency and timing
        self.llm_cooldown_episodes = llm_kwargs.get('llm_cooldown_episodes', 5)
        self.min_episodes_before_llm = llm_kwargs.get('min_episodes_before_llm', 5)

        # Prompt customization
        self.custom_prompt_template = llm_kwargs.get('custom_prompt_template', None)
        self.dataset_name = llm_kwargs.get('dataset_name', 'unknown')
        self.task_description = llm_kwargs.get('task_description', 'hyperparameter optimization')

        # Search space configuration
        self.llm_search_space_mode = llm_kwargs.get('llm_search_space_mode', 'rl_based')  # 'rl_based' or 'llm_guided'
        self.llm_parameter_bounds = llm_kwargs.get('llm_parameter_bounds', None)  # Custom bounds for LLM
        self.llm_parameter_types = llm_kwargs.get('llm_parameter_types', None)  # Custom types for LLM


class LLMEnhancedMAT_HPO_Optimizer(MAT_HPO_Optimizer):
    """
    LLM Enhanced Multi-Agent Transformer HPO Optimizer

    Extends the base MAT_HPO_Optimizer with LLM guidance capabilities.
    Supports multiple mixing strategies including fixed alpha and adaptive triggering.
    """

    def __init__(self,
                 environment,
                 hyperparameter_space,
                 config: LLMEnhancedOptimizationConfig,
                 evaluation_criteria: Optional[ModelSaveCriteria] = None,
                 output_dir: str = None):
        """
        Initialize LLM Enhanced MAT-HPO Optimizer

        Args:
            environment: The environment to optimize
            hyperparameter_space: Hyperparameter search space
            config: LLM enhanced optimization configuration
            evaluation_criteria: Custom evaluation criteria (defaults to SPNV2 criteria)
            output_dir: Custom output directory (overrides default)
        """
        # Use custom output_dir if provided, otherwise use default
        actual_output_dir = output_dir if output_dir is not None else "./mat_hpo_results"
        
        # ✅ 修復：使用評估標準
        if evaluation_criteria is None:
            evaluation_criteria = create_spnv2_criteria()
            
        super().__init__(environment, hyperparameter_space, config, evaluation_criteria, actual_output_dir)

        self.llm_config = config
        self.llm_mixer = None
        
        # ✅ 修復：初始化 metric_names 屬性
        self.metric_names = {}

        if config.enable_llm:
            self._initialize_llm_components()

    def _initialize_llm_components(self):
        """Initialize LLM client and mixer"""
        try:
            # Create LLM client
            llm_client = OllamaLLMClient(
                model_name=self.llm_config.llm_model,
                base_url=self.llm_config.llm_base_url
            )

            # Create enhanced mixer with custom configuration
            mixer_kwargs = {
                'llm_client': llm_client,
                'mixing_strategy': self.llm_config.mixing_strategy,
                'alpha': self.llm_config.alpha,
                'slope_threshold': self.llm_config.slope_threshold,
                'llmpipe_cooldown_period': self.llm_config.llm_cooldown_episodes,
                'dataset_name': self.llm_config.dataset_name,
                'output_dir': getattr(self, 'output_dir', None),
                'dataset_info_csv_path': self.llm_config.dataset_info_csv_path  # 传递数据集信息路径
            }

            # Note: custom_prompt_template, parameter_bounds, parameter_types not supported by EnhancedLLMHyperparameterMixer

            self.llm_mixer = EnhancedLLMHyperparameterMixer(**mixer_kwargs)

            print(f"🤖 LLM Enhanced Optimizer initialized:")
            print(f"   Model: {self.llm_config.llm_model}")
            print(f"   Strategy: {self.llm_config.mixing_strategy}")
            print(f"   Alpha: {self.llm_config.alpha}")
            print(f"   Slope threshold: {self.llm_config.slope_threshold}")
            print(f"   Dataset info path: {self.llm_config.dataset_info_csv_path}")

        except Exception as e:
            print(f"⚠️  LLM initialization failed: {e}")
            print("   Falling back to pure SQDDPG mode")
            self.llm_mixer = None

    def _get_actions(self, states: torch.Tensor, episode: int,
                    current_performance: float = None) -> Tuple[torch.Tensor, Dict]:
        """
        Get actions with optional LLM guidance

        Args:
            states: Current states for all agents
            episode: Current episode number
            current_performance: Current performance metric for LLM decision

        Returns:
            Tuple of (actions, llm_info)
        """
        actions, llm_info = super()._get_actions(states, episode)

        # Try LLM guidance if available and conditions are met
        if self.llm_mixer is not None and current_performance is not None:
            try:
                # Get Q-values for uncertainty detection (if available)
                q_values = None
                if hasattr(self.agents[0], 'critic') and hasattr(self.agents[0].critic, 'Q1'):
                    with torch.no_grad():
                        q_values = self.agents[0].critic.Q1(states, actions)

                # Check if LLM should intervene
                llm_actions, llm_usage_info = self.llm_mixer.get_mixed_actions(
                    rl_actions=actions,
                    states=states,
                    episode=episode,
                    current_performance=current_performance,
                    q_values=q_values
                )

                if llm_actions is not None:
                    actions = llm_actions
                    llm_info.update(llm_usage_info)

            except Exception as e:
                print(f"⚠️  LLM guidance failed: {e}")

        return actions, llm_info

    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization with LLM enhancement by mixing hyperparameters before evaluation.

        This overrides the base optimize loop to inject LLM suggestions via
        EnhancedLLMHyperparameterMixer.get_mixed_hyperparameters().

        Returns:
            Optimization results with optional LLM statistics
        """
        self.logger.info(f"🚀 Starting LLM Enhanced MAT-HPO optimization for {self.config.max_steps} steps")
        self.logger.info(f"LLM enabled: {self.llm_config.enable_llm}")

        # Get parameter count in a compatible way
        if hasattr(self.hyperparameter_space, 'parameters'):
            param_count = len(self.hyperparameter_space.parameters)
        else:
            param_count = sum(self.hyperparameter_space.agent_dims)
        self.logger.info(f"📊 Search space: {param_count} parameters across 3 agents")

        start_time = time.time()

        # Initialize environment
        state = self.environment.reset().to(self.device)

        for step in range(self.config.max_steps):
            self.current_step = step
            step_start_time = time.time()

            # Generate actions using SQDDPG and translate to RL hyperparameters
            actions = self.sqddpg.policy(state)
            selected_actions = self.sqddpg.select_action(actions)
            rl_hyperparams = self._extract_hyperparameters(selected_actions)

            # Optionally mix with LLM suggestions at the hyperparameter level
            final_hyperparams = rl_hyperparams
            decision_source = "RL"

            if self.llm_mixer is not None and self.llm_config.enable_llm:
                try:
                    # Build lightweight dataset info for prompts
                    dataset_name = getattr(self.environment, 'dataset', 'Unknown')
                    dataset_info = {'name': str(dataset_name)}

                    # Current performance context (best so far if available)
                    current_perf = None
                    if self.best_reward != float('-inf'):
                        current_perf = {
                            'f1': float(self.best_f1),
                            'auc': float(self.best_auc),
                            'gmean': float(self.best_gmean)
                        }

                    final_hyperparams, decision_source = self.llm_mixer.get_mixed_hyperparameters(
                        hyperparameter_space=self.hyperparameter_space,
                        rl_hyperparams=rl_hyperparams,
                        dataset_info=dataset_info,
                        step=step,
                        current_performance=current_perf
                    )
                except Exception as e:
                    print(f"⚠️  LLM mixing failed at step {step}: {e}. Falling back to RL hyperparameters.")
                    final_hyperparams = rl_hyperparams
                    decision_source = "RL (fallback)"

            # Display the (possibly mixed) hyperparameters for this step
            self._display_step_hyperparameters(step, final_hyperparams)

            # Evaluate in environment
            reward, metrics, done = self.environment.step(final_hyperparams)
            
            # Extract metrics for logging
            f1 = metrics.get('val_f1', metrics.get('f1', 0.0))
            auc = metrics.get('val_auc', metrics.get('auc', 0.0))
            gmean = metrics.get('val_gmean', metrics.get('gmean', 0.0))

            step_time = time.time() - step_start_time

            # Log step results
            self.logger.log_step(step, reward, metrics, step_time, final_hyperparams)

            # Update LLM training history (non-fatal if it fails)
            if self.llm_mixer is not None and self.llm_config.enable_llm:
                try:
                    self.llm_mixer.update_history(
                        metrics={'f1': f1, 'auc': auc, 'gmean': gmean},
                        hyperparams=final_hyperparams,
                        step=step
                    )
                except Exception as e:
                    print(f"⚠️  Failed to update LLM history at step {step}: {e}")

            # ✅ 修復：使用靈活的評估器（繼承自父類）
            should_save = self.evaluator.evaluate(metrics, step, final_hyperparams)
            
            if should_save:
                self._save_best_model(step)  # 傳入正確的 step
                self._save_rl_model_input(state)  # Save current state as RL_model_input.pt
                
                # 更新內部追蹤變量（向後兼容）
                self.best_reward = self.evaluator.best_score
                self.best_hyperparams = self.evaluator.best_hyperparams
                self.best_metrics = self.evaluator.best_metrics
                self.best_step = self.evaluator.best_step
                self.best_f1, self.best_auc, self.best_gmean = f1, auc, gmean
                
                if self.metric_names:
                    metric1_name = self.metric_names.get('f1', 'F1')
                    metric2_name = self.metric_names.get('auc', 'AUC')
                    metric3_name = self.metric_names.get('gmean', 'G-mean')
                    self.logger.info(f"🎯 New best model at step {step}! "
                                   f"{metric1_name}={f1:.4f}, {metric2_name}={auc:.4f}, {metric3_name}={gmean:.4f} (via {decision_source})")
                else:
                    self.logger.info(
                        f"🎯 New best model at step {step}! F1={f1:.4f}, AUC={auc:.4f}, G-mean={gmean:.4f} (via {decision_source})"
                    )

            # Update replay buffer with reward signal
            reward_array = np.array([reward] * len(self.action_optimizers))
            transition = Transition(
                state.cpu().numpy(),
                selected_actions.detach().cpu().numpy(),
                reward_array
            )
            self.replay_buffer.add_experience(transition)

            # Update agents if enough samples
            if (len(self.replay_buffer) >= self.config.batch_size and 
                step % self.config.behaviour_update_freq == 0):
                self._update_agents()

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

        # Add LLM statistics to results if available
        if self.llm_mixer is not None and self.llm_config.enable_llm:
            try:
                llm_stats = self.llm_mixer.get_usage_stats()
                results['llm_statistics'] = llm_stats

                # Print LLM usage summary
                print(f"\n📊 LLM Usage Statistics:")
                print(f"   Total decisions: {llm_stats.get('total', 0)}")
                print(f"   LLM interventions: {llm_stats.get('llm_count', 0)} ({llm_stats.get('llm_pct', 0):.1f}%)")
                print(f"   RL decisions: {llm_stats.get('rl_count', 0)} ({llm_stats.get('rl_pct', 0):.1f}%)")

                if 'adaptive_alpha_stats' in llm_stats:
                    aa = llm_stats['adaptive_alpha_stats']
                    print(f"   Adaptive alpha adjustments: {aa.get('total_adjustments', 0)}")

                if 'llmpipe_stats' in llm_stats:
                    lp = llm_stats['llmpipe_stats']
                    print(f"   LLaPipe triggers: {lp.get('total_triggers', 0)}")
            except Exception as e:
                print(f"⚠️  Failed to collect LLM statistics: {e}")

        self.logger.info(f"Optimization completed in {total_time:.2f} seconds")
        self.logger.info(f"Best hyperparameters: {self.best_hyperparams}")
        self.logger.info(f"Best performance: F1={self.best_f1:.4f}, AUC={self.best_auc:.4f}, G-mean={self.best_gmean:.4f}")

        return results

    def get_best_performance(self) -> Dict[str, Any]:
        """Get best performance including LLM statistics"""
        performance = super().get_best_performance()

        if self.llm_mixer is not None:
            performance['llm_statistics'] = self.llm_mixer.get_usage_stats()

        return performance


# Convenience functions for easy LLM configuration
class LLMConfigs:
    """Pre-configured LLM settings for common scenarios"""
    
    @staticmethod
    def fixed_alpha(dataset_name: str = "unknown", alpha: float = 0.3, 
                   model: str = "llama3.2:3b") -> LLMEnhancedOptimizationConfig:
        """Fixed alpha LLM mixing strategy"""
        return LLMEnhancedOptimizationConfig(
            enable_llm=True,
            llm_model=model,
            mixing_strategy='fixed_alpha',
            alpha=alpha,
            dataset_name=dataset_name,
            task_description='hyperparameter optimization'
        )
    
    @staticmethod
    def adaptive_alpha(dataset_name: str = "unknown", slope_threshold: float = 0.01,
                      model: str = "llama3.2:3b") -> LLMEnhancedOptimizationConfig:
        """Adaptive alpha LLM mixing strategy"""
        return LLMEnhancedOptimizationConfig(
            enable_llm=True,
            llm_model=model,
            mixing_strategy='adaptive_alpha',
            slope_threshold=slope_threshold,
            dataset_name=dataset_name,
            task_description='hyperparameter optimization'
        )
    
    @staticmethod
    def llmpipe(dataset_name: str = "unknown", slope_threshold: float = 0.01,
                model: str = "llama3.2:3b") -> LLMEnhancedOptimizationConfig:
        """LLaPipe adaptive triggering strategy"""
        return LLMEnhancedOptimizationConfig(
            enable_llm=True,
            llm_model=model,
            mixing_strategy='llmpipe',
            slope_threshold=slope_threshold,
            dataset_name=dataset_name,
            task_description='hyperparameter optimization'
        )
    
    @staticmethod
    def hybrid(dataset_name: str = "unknown", alpha: float = 0.3, 
               slope_threshold: float = 0.01, model: str = "llama3.2:3b") -> LLMEnhancedOptimizationConfig:
        """Hybrid strategy combining adaptive alpha and LLaPipe"""
        return LLMEnhancedOptimizationConfig(
            enable_llm=True,
            llm_model=model,
            mixing_strategy='hybrid',
            alpha=alpha,
            slope_threshold=slope_threshold,
            dataset_name=dataset_name,
            task_description='hyperparameter optimization'
        )
    
    @staticmethod
    def custom(dataset_name: str, task_description: str, 
               custom_prompt: str = None, parameter_bounds: dict = None,
               parameter_types: dict = None, **llm_kwargs) -> LLMEnhancedOptimizationConfig:
        """Custom LLM configuration with full control"""
        config = LLMEnhancedOptimizationConfig(
            enable_llm=True,
            dataset_name=dataset_name,
            task_description=task_description,
            custom_prompt_template=custom_prompt,
            llm_parameter_bounds=parameter_bounds,
            llm_parameter_types=parameter_types,
            **llm_kwargs
        )
        return config