"""
Enhanced LLM Hyperparameter Mixer with user-configurable LLM clients

IMPORTANT CLARIFICATION:
- "llmpipe" strategy = Original LLaPipe paper implementation (adaptive trigger with uncertainty + stagnation detection)
- "adaptive_alpha" strategy = Additional enhancement beyond LLaPipe (continuous alpha adjustment)
- LLaPipe already includes regression-based slope monitoring for stagnation detection
- This implementation adds continuous probability adjustment as an extension

Provides advanced mixing strategies with support for custom LLM clients,
LLaPipe adaptive triggering, and continuous alpha adjustment mechanisms.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import time
from sklearn.linear_model import LinearRegression

from .base_llm_client import BaseLLMClient
from .ollama_client import OllamaLLMClient
from .adaptive_advisor import LLaPipeAdaptiveAdvisor, PerformanceMetricCalculator
from .conversation_logger import LLMConversationLogger


class AdaptiveAlphaController:
    """
    Advanced adaptive alpha controller with regression-based slope monitoring

    Dynamically adjusts the LLM usage probability based on:
    1. Performance improvement trends (regression slope)
    2. Uncertainty levels in RL agent decisions
    3. Training stability indicators
    4. User-defined thresholds and constraints
    """

    def __init__(self,
                 initial_alpha: float = 0.3,
                 min_alpha: float = 0.1,
                 max_alpha: float = 0.8,
                 slope_threshold: float = 0.01,
                 performance_window: int = 10,
                 alpha_adjustment_rate: float = 0.1,
                 stability_threshold: float = 0.02):
        """
        Args:
            initial_alpha: Starting alpha value
            min_alpha: Minimum allowed alpha value
            max_alpha: Maximum allowed alpha value
            slope_threshold: Minimum performance slope to maintain RL control
            performance_window: Number of recent episodes for slope calculation
            alpha_adjustment_rate: Rate of alpha adjustment (0.0-1.0)
            stability_threshold: Threshold for performance stability detection
        """
        self.initial_alpha = initial_alpha
        self.current_alpha = initial_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.slope_threshold = slope_threshold
        self.performance_window = performance_window
        self.alpha_adjustment_rate = alpha_adjustment_rate
        self.stability_threshold = stability_threshold

        # Performance tracking
        self.performance_history = deque(maxlen=performance_window)
        self.slope_history = []
        self.alpha_history = []

        # Statistics
        self.total_adjustments = 0
        self.increase_adjustments = 0
        self.decrease_adjustments = 0

        print(f"🎛️ Adaptive Alpha Controller initialized:")
        print(f"   Initial alpha: {initial_alpha}")
        print(f"   Alpha range: [{min_alpha}, {max_alpha}]")
        print(f"   Slope threshold: {slope_threshold}")
        print(f"   Performance window: {performance_window}")

    def update_performance(self, current_performance: float, episode: int) -> float:
        """
        Update performance history and adjust alpha based on trends

        Args:
            current_performance: Current episode performance
            episode: Current episode number

        Returns:
            Updated alpha value
        """
        self.performance_history.append({
            'performance': current_performance,
            'episode': episode,
            'timestamp': time.time()
        })

        # Need at least 3 data points for meaningful slope analysis
        if len(self.performance_history) < 3:
            return self.current_alpha

        # Calculate performance slope
        episodes = np.array([p['episode'] for p in self.performance_history]).reshape(-1, 1)
        performances = np.array([p['performance'] for p in self.performance_history])

        try:
            reg = LinearRegression()
            reg.fit(episodes, performances)
            current_slope = reg.coef_[0]
            r_squared = reg.score(episodes, performances)

            self.slope_history.append(current_slope)

            # Adjust alpha based on slope and stability
            previous_alpha = self.current_alpha
            self._adjust_alpha(current_slope, r_squared, current_performance)

            # Track adjustment statistics
            if self.current_alpha != previous_alpha:
                self.total_adjustments += 1
                if self.current_alpha > previous_alpha:
                    self.increase_adjustments += 1
                else:
                    self.decrease_adjustments += 1

            self.alpha_history.append(self.current_alpha)

            return self.current_alpha

        except Exception as e:
            print(f"⚠️ Alpha adjustment failed: {e}")
            return self.current_alpha

    def _adjust_alpha(self, slope: float, r_squared: float, current_performance: float):
        """Adjust alpha based on performance analysis"""
        alpha_change = 0.0

        # Slope-based adjustment
        if slope < self.slope_threshold:
            # Poor learning progress -> increase LLM intervention
            alpha_change += self.alpha_adjustment_rate * (self.slope_threshold - slope)
        elif slope > self.slope_threshold * 2:
            # Good learning progress -> reduce LLM intervention
            alpha_change -= self.alpha_adjustment_rate * 0.5

        # Stability-based adjustment
        if len(self.performance_history) >= 5:
            recent_performances = [p['performance'] for p in list(self.performance_history)[-5:]]
            stability = np.std(recent_performances)

            if stability > self.stability_threshold:
                # High instability -> increase LLM guidance
                alpha_change += self.alpha_adjustment_rate * 0.3
            else:
                # Stable learning -> trust RL more
                alpha_change -= self.alpha_adjustment_rate * 0.2

        # R-squared based adjustment (trend reliability)
        if r_squared < 0.3:
            # Unreliable trend -> increase LLM intervention
            alpha_change += self.alpha_adjustment_rate * 0.2

        # Apply adjustment
        new_alpha = self.current_alpha + alpha_change
        self.current_alpha = max(self.min_alpha, min(self.max_alpha, new_alpha))

        # Debug output
        if alpha_change != 0:
            direction = "↗️" if alpha_change > 0 else "↘️"
            print(f"{direction} Alpha adjusted: {self.current_alpha:.3f} "
                  f"(slope: {slope:.6f}, stability: {np.std([p['performance'] for p in list(self.performance_history)[-5:]]):.4f})")

    def get_statistics(self) -> Dict:
        """Get adaptive alpha controller statistics"""
        stats = {
            'current_alpha': self.current_alpha,
            'initial_alpha': self.initial_alpha,
            'alpha_range': [self.min_alpha, self.max_alpha],
            'total_adjustments': self.total_adjustments,
            'increase_adjustments': self.increase_adjustments,
            'decrease_adjustments': self.decrease_adjustments,
            'slope_threshold': self.slope_threshold,
            'performance_window': self.performance_window
        }

        if self.slope_history:
            stats.update({
                'avg_slope': np.mean(self.slope_history),
                'slope_std': np.std(self.slope_history),
                'recent_slope': self.slope_history[-1] if self.slope_history else 0
            })

        if self.alpha_history:
            stats.update({
                'alpha_variance': np.var(self.alpha_history),
                'alpha_trend': np.mean(np.diff(self.alpha_history)) if len(self.alpha_history) > 1 else 0
            })

        return stats


class EnhancedLLMHyperparameterMixer:
    """
    Enhanced LLM Hyperparameter Mixer with user-configurable LLM clients

    Supports multiple LLM mixing strategies:
    1. Fixed alpha (constant probability)
    2. Adaptive alpha (regression-based slope monitoring)
    3. LLaPipe-style adaptive triggering (uncertainty + stagnation detection)
    4. Hybrid approaches combining multiple strategies

    Key enhancements:
    - User-configurable LLM clients (OpenAI, Anthropic, Ollama, custom)
    - Advanced performance slope monitoring with regression analysis
    - Configurable thresholds for all decision mechanisms
    - Comprehensive logging and statistics
    - Fallback mechanisms for robust operation
    """

    def __init__(self,
                 llm_client: Optional[BaseLLMClient] = None,
                 alpha: float = 0.3,
                 dataset_name: str = "Unknown",
                 output_dir: str = None,
                 dataset_info_csv_path: str = "./Datasets_info.csv",
                 # Strategy selection
                 mixing_strategy: str = "fixed_alpha",  # "fixed_alpha", "adaptive_alpha", "llmpipe", "hybrid"
                 # Adaptive alpha parameters
                 alpha_min: float = 0.1,
                 alpha_max: float = 0.8,
                 slope_threshold: float = 0.01,
                 alpha_adjustment_rate: float = 0.1,
                 # LLaPipe parameters
                 llmpipe_slope_threshold: float = 0.01,
                 llmpipe_uncertainty_threshold: float = 0.1,
                 llmpipe_cooldown_period: int = 5,
                 # Performance monitoring
                 performance_window: int = 10,
                 stability_threshold: float = 0.02):
        """
        Initialize Enhanced LLM Hyperparameter Mixer

        Args:
            llm_client: Custom LLM client (if None, uses OllamaLLMClient)
            alpha: Base alpha for fixed strategy or initial alpha for adaptive
            dataset_name: Dataset name for logging
            output_dir: Output directory for logs
            dataset_info_csv_path: Path to dataset information CSV file
            mixing_strategy: Strategy to use ("fixed_alpha", "adaptive_alpha", "llmpipe", "hybrid")
            alpha_min/max: Alpha bounds for adaptive strategy
            slope_threshold: Performance slope threshold
            alpha_adjustment_rate: Rate of alpha adjustments
            llmpipe_*: LLaPipe-specific parameters
            performance_window: Window size for performance analysis
            stability_threshold: Threshold for stability detection
        """
        self.dataset_name = dataset_name
        self.mixing_strategy = mixing_strategy
        self.training_history = []
        self.rl_decision_count = 0
        self.llm_decision_count = 0

        # Initialize LLM client
        if llm_client is not None:
            self.llm_client = llm_client
            print(f"🤖 Using custom LLM client: {llm_client.__class__.__name__}")
        else:
            self.llm_client = OllamaLLMClient()
            print(f"🤖 Using default Ollama LLM client")

        # Initialize strategy-specific components
        self.alpha = alpha
        self.alpha_controller = None
        self.adaptive_advisor = None

        if mixing_strategy == "fixed_alpha":
            print(f"📊 Fixed alpha strategy: {alpha}")

        elif mixing_strategy == "adaptive_alpha":
            # Note: This is an additional enhancement beyond LLaPipe
            # LLaPipe already includes slope monitoring for triggering
            # This adds continuous alpha adjustment based on performance trends
            self.alpha_controller = AdaptiveAlphaController(
                initial_alpha=alpha,
                min_alpha=alpha_min,
                max_alpha=alpha_max,
                slope_threshold=slope_threshold,
                performance_window=performance_window,
                alpha_adjustment_rate=alpha_adjustment_rate,
                stability_threshold=stability_threshold
            )
            print(f"🎛️ Continuous alpha adjustment strategy: [{alpha_min}, {alpha_max}]")
            print(f"   Note: This extends beyond LLaPipe's discrete triggering")

        elif mixing_strategy == "llmpipe":
            self.adaptive_advisor = LLaPipeAdaptiveAdvisor(
                slope_threshold=llmpipe_slope_threshold,
                buffer_size=performance_window,
                cooldown_period=llmpipe_cooldown_period,
                uncertainty_threshold=llmpipe_uncertainty_threshold,
                min_episodes_before_trigger=5
            )
            print(f"🎯 LLaPipe adaptive triggering (uncertainty + stagnation detection)")

        elif mixing_strategy == "hybrid":
            # Combine adaptive alpha with LLaPipe uncertainty detection
            self.alpha_controller = AdaptiveAlphaController(
                initial_alpha=alpha,
                min_alpha=alpha_min,
                max_alpha=alpha_max,
                slope_threshold=slope_threshold,
                performance_window=performance_window,
                alpha_adjustment_rate=alpha_adjustment_rate,
                stability_threshold=stability_threshold
            )
            self.adaptive_advisor = LLaPipeAdaptiveAdvisor(
                slope_threshold=llmpipe_slope_threshold,
                buffer_size=performance_window,
                cooldown_period=llmpipe_cooldown_period,
                uncertainty_threshold=llmpipe_uncertainty_threshold,
                min_episodes_before_trigger=5
            )
            print(f"🔄 Hybrid strategy: Continuous alpha adjustment + LLaPipe triggering")

        else:
            raise ValueError(f"Unknown mixing strategy: {mixing_strategy}")

        # Random generator for probabilistic decisions
        self.decision_rng = np.random.RandomState(42)

        # Initialize conversation logger
        strategy_str = f"{mixing_strategy}_{alpha}" if mixing_strategy == "fixed_alpha" else mixing_strategy
        self.conversation_logger = LLMConversationLogger(dataset_name, strategy_str, output_dir=output_dir)

        # Set logger in LLM client
        if hasattr(self.llm_client, 'conversation_logger'):
            self.llm_client.conversation_logger = self.conversation_logger

        # Store dataset info CSV path for use in prompt generation
        self.dataset_info_csv_path = dataset_info_csv_path

    def should_use_llm(self,
                      current_performance: Optional[Dict] = None,
                      q_values: Optional[torch.Tensor] = None,
                      action_probs: Optional[torch.Tensor] = None,
                      episode: int = 0) -> Tuple[bool, Dict]:
        """
        Advanced decision function for LLM usage

        Combines multiple decision strategies based on the configured mixing strategy.

        Args:
            current_performance: Current performance metrics
            q_values: Q-values for uncertainty detection
            action_probs: Action probabilities for uncertainty detection
            episode: Current episode number

        Returns:
            Tuple of (should_use_llm, decision_info)
        """
        decision_info = {
            'strategy': self.mixing_strategy,
            'episode': episode,
            'method': None,
            'reason': None
        }

        if self.mixing_strategy == "fixed_alpha":
            return self._fixed_alpha_decision(decision_info)

        elif self.mixing_strategy == "adaptive_alpha":
            return self._adaptive_alpha_decision(current_performance, episode, decision_info)

        elif self.mixing_strategy == "llmpipe":
            return self._llmpipe_decision(current_performance, q_values, action_probs, decision_info)

        elif self.mixing_strategy == "hybrid":
            return self._hybrid_decision(current_performance, q_values, action_probs, episode, decision_info)

        else:
            # Fallback to fixed alpha
            return self._fixed_alpha_decision(decision_info)

    def _fixed_alpha_decision(self, decision_info: Dict) -> Tuple[bool, Dict]:
        """Fixed alpha probability decision"""
        use_llm = self.decision_rng.random() < self.alpha
        decision_info.update({
            'method': 'fixed_alpha',
            'alpha': self.alpha,
            'random_draw': use_llm
        })
        return use_llm, decision_info

    def _adaptive_alpha_decision(self, current_performance: Optional[Dict], episode: int,
                               decision_info: Dict) -> Tuple[bool, Dict]:
        """Adaptive alpha decision with regression-based slope monitoring"""
        if current_performance is None or self.alpha_controller is None:
            return self._fixed_alpha_decision(decision_info)

        # Calculate unified performance metric
        unified_metric = PerformanceMetricCalculator.adaptive_metric_selection(
            current_performance.get('f1', 0),
            current_performance.get('auc', 0),
            current_performance.get('gmean', 0)
        )

        # Update alpha based on performance trends
        current_alpha = self.alpha_controller.update_performance(unified_metric, episode)

        # Make probabilistic decision with current alpha
        use_llm = self.decision_rng.random() < current_alpha

        decision_info.update({
            'method': 'adaptive_alpha',
            'current_alpha': current_alpha,
            'unified_metric': unified_metric,
            'random_draw': use_llm
        })

        # Add slope information if available
        if self.alpha_controller.slope_history:
            decision_info['current_slope'] = self.alpha_controller.slope_history[-1]

        return use_llm, decision_info

    def _llmpipe_decision(self, current_performance: Optional[Dict],
                         q_values: Optional[torch.Tensor],
                         action_probs: Optional[torch.Tensor],
                         decision_info: Dict) -> Tuple[bool, Dict]:
        """LLaPipe adaptive triggering decision"""
        if current_performance is None or self.adaptive_advisor is None:
            decision_info.update({'method': 'llmpipe', 'reason': 'no_performance_data'})
            return False, decision_info

        # Calculate unified performance metric
        unified_metric = PerformanceMetricCalculator.adaptive_metric_selection(
            current_performance.get('f1', 0),
            current_performance.get('auc', 0),
            current_performance.get('gmean', 0)
        )

        # Check if LLM should be triggered
        should_trigger, trigger_info = self.adaptive_advisor.should_trigger_llm(
            current_performance=unified_metric,
            q_values=q_values,
            action_probs=action_probs
        )

        decision_info.update({
            'method': 'llmpipe',
            'unified_metric': unified_metric,
            **trigger_info
        })

        return should_trigger, decision_info

    def _hybrid_decision(self, current_performance: Optional[Dict],
                        q_values: Optional[torch.Tensor],
                        action_probs: Optional[torch.Tensor],
                        episode: int,
                        decision_info: Dict) -> Tuple[bool, Dict]:
        """Hybrid decision combining adaptive alpha with uncertainty detection"""
        if current_performance is None:
            return self._fixed_alpha_decision(decision_info)

        # Get LLaPipe triggering decision
        llmpipe_trigger, llmpipe_info = self._llmpipe_decision(
            current_performance, q_values, action_probs, decision_info.copy()
        )

        # Get adaptive alpha decision
        adaptive_trigger, adaptive_info = self._adaptive_alpha_decision(
            current_performance, episode, decision_info.copy()
        )

        # Hybrid logic: Use LLM if either condition is met, but weight by alpha
        if llmpipe_trigger:
            # LLaPipe triggered - always use LLM
            final_decision = True
            decision_info.update({
                'method': 'hybrid_llmpipe_trigger',
                'llmpipe_reason': llmpipe_info.get('trigger_reason'),
                'adaptive_alpha': adaptive_info.get('current_alpha')
            })
        else:
            # No LLaPipe trigger - use adaptive alpha probability
            final_decision = adaptive_trigger
            decision_info.update({
                'method': 'hybrid_adaptive_alpha',
                'llmpipe_reason': llmpipe_info.get('trigger_reason'),
                'adaptive_alpha': adaptive_info.get('current_alpha'),
                'random_draw': adaptive_trigger
            })

        # Include information from both strategies
        decision_info.update({
            'unified_metric': adaptive_info.get('unified_metric'),
            'current_slope': adaptive_info.get('current_slope')
        })

        return final_decision, decision_info

    def get_mixed_hyperparameters(self,
                                hyperparameter_space,
                                rl_hyperparams: Dict[str, Any],
                                dataset_info: Dict,
                                step: int = 0,
                                current_performance: Optional[Dict] = None,
                                q_values: Optional[torch.Tensor] = None) -> Tuple[Dict[str, Any], str]:
        """
        Get mixed hyperparameters from RL and LLM

        Args:
            hyperparameter_space: HyperparameterSpace object
            rl_hyperparams: Hyperparameters from RL agent
            dataset_info: Dataset information for LLM
            step: Current optimization step
            current_performance: Recent performance for adaptive strategies
            q_values: Q-values for uncertainty detection

        Returns:
            Tuple of (final_hyperparams, decision_source)
        """
        # Set current step for logging
        if hasattr(self.llm_client, 'current_step'):
            self.llm_client.current_step = step

        # Decide whether to use LLM
        use_llm, decision_info = self.should_use_llm(current_performance, q_values, episode=step)

        if use_llm:
            self.llm_decision_count += 1

            try:
                llm_hyperparams = self.llm_client.generate_hyperparameters(
                    hyperparameter_space=hyperparameter_space,
                    dataset_info=dataset_info,
                    training_history=self.training_history
                )

                # Enhanced decision source with strategy info
                decision_source = f"LLM ({self.mixing_strategy})"
                self._print_llm_decision(decision_info, step)

                return llm_hyperparams, decision_source

            except Exception as e:
                print(f"🚨 LLM generation failed: {e}")
                self.rl_decision_count += 1
                return rl_hyperparams, "RL (LLM fallback)"

        else:
            self.rl_decision_count += 1
            decision_source = f"RL ({self.mixing_strategy})"
            self._print_rl_decision(decision_info, step)

            return rl_hyperparams, decision_source

    def _print_llm_decision(self, decision_info: Dict, step: int):
        """Print detailed LLM decision information"""
        method = decision_info.get('method', 'unknown')

        if method == 'fixed_alpha':
            print(f"🤖 LLM decision (fixed α={decision_info.get('alpha', 0):.3f})")

        elif method == 'adaptive_alpha':
            alpha = decision_info.get('current_alpha', 0)
            slope = decision_info.get('current_slope', 0)
            print(f"🤖 LLM decision (adaptive α={alpha:.3f}, slope={slope:.6f})")

        elif method == 'llmpipe':
            reason = decision_info.get('trigger_reason', 'unknown')
            slope = decision_info.get('learning_slope', 0)
            if slope is not None:
                print(f"🤖 LLM triggered (LLaPipe: {reason}, slope={slope:.6f})")
            else:
                print(f"🤖 LLM triggered (LLaPipe: {reason})")

        elif method.startswith('hybrid'):
            if 'llmpipe_trigger' in method:
                reason = decision_info.get('llmpipe_reason', 'unknown')
                alpha = decision_info.get('adaptive_alpha', 0)
                print(f"🤖 LLM triggered (Hybrid: LLaPipe {reason}, α={alpha:.3f})")
            else:
                alpha = decision_info.get('adaptive_alpha', 0)
                print(f"🤖 LLM decision (Hybrid: α={alpha:.3f})")

    def _print_rl_decision(self, decision_info: Dict, step: int):
        """Print detailed RL decision information"""
        method = decision_info.get('method', 'unknown')

        if method == 'fixed_alpha':
            print(f"🎯 RL decision (fixed α={decision_info.get('alpha', 0):.3f})")

        elif method == 'adaptive_alpha':
            alpha = decision_info.get('current_alpha', 0)
            slope = decision_info.get('current_slope', 0)
            print(f"🎯 RL decision (adaptive α={alpha:.3f}, slope={slope:.6f})")

        elif method == 'llmpipe':
            reason = decision_info.get('trigger_reason', 'unknown')
            print(f"🎯 RL continuing (LLaPipe: {reason})")

        elif method.startswith('hybrid'):
            reason = decision_info.get('llmpipe_reason', 'unknown')
            alpha = decision_info.get('adaptive_alpha', 0)
            print(f"🎯 RL decision (Hybrid: {reason}, α={alpha:.3f})")

    def update_history(self, metrics: Dict[str, float], hyperparams: Dict[str, Any], step: int = 0):
        """Update training history"""
        self.training_history.append({
            'metrics': metrics.copy(),
            'hyperparams': hyperparams.copy(),
            'step': step
        })

        # Keep only recent history
        if len(self.training_history) > 15:
            self.training_history = self.training_history[-15:]

    def get_usage_stats(self) -> Dict:
        """Get comprehensive usage statistics"""
        total = self.rl_decision_count + self.llm_decision_count
        base_stats = {
            'rl_count': self.rl_decision_count,
            'llm_count': self.llm_decision_count,
            'rl_pct': (self.rl_decision_count / max(total, 1)) * 100,
            'llm_pct': (self.llm_decision_count / max(total, 1)) * 100,
            'total': total,
            'mixing_strategy': self.mixing_strategy
        }

        # Add strategy-specific statistics
        if self.alpha_controller:
            base_stats.update({
                'adaptive_alpha_stats': self.alpha_controller.get_statistics()
            })

        if self.adaptive_advisor:
            base_stats.update({
                'llmpipe_stats': self.adaptive_advisor.get_statistics()
            })

        if hasattr(self.llm_client, 'get_statistics'):
            base_stats.update({
                'llm_client_stats': self.llm_client.get_statistics()
            })

        return base_stats

    def generate_hyperparameters(self, 
                                dataset_info: Dict,
                                training_history: List[Dict],
                                current_state: torch.Tensor,
                                step: int = 0,
                                current_performance: Optional[Dict] = None,
                                q_values: Optional[torch.Tensor] = None,
                                action_probs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, str]:
        """
        Generate hyperparameters using the configured mixing strategy
        
        Args:
            dataset_info: Dataset characteristics
            training_history: Previous training results
            current_state: Current environment state
            step: Current optimization step
            current_performance: Current performance metrics
            q_values: Q-values for uncertainty detection
            action_probs: Action probabilities for uncertainty detection
            
        Returns:
            Tuple of (hyperparameters_tensor, decision_source)
        """
        # Decide whether to use LLM or RL
        use_llm, decision_info = self.should_use_llm(
            current_performance=current_performance,
            q_values=q_values,
            action_probs=action_probs,
            episode=step
        )
        
        if use_llm:
            return self._generate_llm_hyperparameters(
                dataset_info, training_history, current_state, step, decision_info
            )
        else:
            return self._generate_rl_hyperparameters(
                current_state, decision_info
            )
    
    def _generate_llm_hyperparameters(self, 
                                     dataset_info: Dict,
                                     training_history: List[Dict],
                                     current_state: torch.Tensor,
                                     step: int,
                                     decision_info: Dict) -> Tuple[torch.Tensor, str]:
        """Generate hyperparameters using LLM"""
        try:
            # Generate hyperparameters using LLM client
            agent0_params, agent1_params, agent2_params = self.llm_client.generate_all_hyperparameters(
                dataset_info=dataset_info,
                training_history=training_history,
                current_state=current_state
            )
            
            # Convert to tensor format
            hyperparams_tensor = self._populate_parameters_dict(
                agent0_params, agent1_params, agent2_params, dataset_info
            )
            
            # Update decision tracking
            self.llm_decision_count += 1
            decision_source = f"LLM_{decision_info.get('method', 'unknown')}"
            
            return hyperparams_tensor, decision_source
            
        except Exception as e:
            print(f"🚨 LLM generation failed: {e}")
            # Fallback to RL
            return self._generate_rl_hyperparameters(current_state, decision_info)
    
    def _generate_rl_hyperparameters(self, 
                                    current_state: torch.Tensor,
                                    decision_info: Dict) -> Tuple[torch.Tensor, str]:
        """Generate hyperparameters using RL (return current state as-is)"""
        self.rl_decision_count += 1
        decision_source = f"RL_{decision_info.get('method', 'unknown')}"
        return current_state, decision_source
    
    def _populate_parameters_dict(self, 
                                 agent0_params: List[float],
                                 agent1_params: List[float], 
                                 agent2_params: List[float],
                                 dataset_info: Dict) -> torch.Tensor:
        """
        Populate parameters dictionary and convert to tensor format
        
        Args:
            agent0_params: Class weights from agent 0
            agent1_params: Architecture parameters from agent 1
            agent2_params: Training parameters from agent 2
            dataset_info: Dataset information
            
        Returns:
            Tensor representation of hyperparameters
        """
        # Create parameters dictionary
        params_dict = {}
        
        # Agent 0: Class weights
        num_classes = dataset_info.get('num_classes', len(agent0_params))
        for i in range(num_classes):
            if i < len(agent0_params):
                params_dict[f'class_weight_{i}'] = agent0_params[i]
            else:
                params_dict[f'class_weight_{i}'] = 1.0  # Default weight
        
        # Agent 1: Architecture parameters
        if len(agent1_params) >= 1:
            params_dict['hidden_size'] = int(agent1_params[0])
        else:
            params_dict['hidden_size'] = 256  # Default
        
        # Agent 2: Training parameters
        if len(agent2_params) >= 2:
            params_dict['batch_size'] = int(agent2_params[0])
            params_dict['learning_rate'] = float(agent2_params[1])
        else:
            params_dict['batch_size'] = 32  # Default
            params_dict['learning_rate'] = 1e-4  # Default
        
        # Convert to tensor format (matching the expected format)
        # This should match the format expected by the environment
        tensor_params = []
        
        # Add class weights
        for i in range(num_classes):
            tensor_params.append(params_dict[f'class_weight_{i}'])
        
        # Add architecture parameters
        tensor_params.append(params_dict['hidden_size'])
        
        # Add training parameters
        tensor_params.append(params_dict['batch_size'])
        tensor_params.append(params_dict['learning_rate'])
        
        return torch.tensor(tensor_params, dtype=torch.float32)