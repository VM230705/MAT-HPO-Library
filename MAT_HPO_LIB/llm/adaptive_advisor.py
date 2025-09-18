"""
Adaptive Advisor implementation based on LLaPipe paper methodology
Adapted for MAT_HPO_LIB from MAT_HPO_LLM
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
import time
from sklearn.linear_model import LinearRegression


class PerformanceMetricCalculator:
    """Helper class to calculate unified performance metrics for triggering"""
    
    @staticmethod
    def calculate_unified_metric(f1: float, auc: float, gmean: float, 
                               weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)) -> float:
        """
        Calculate weighted unified performance metric
        
        Args:
            f1, auc, gmean: Individual metrics
            weights: Weights for (f1, auc, gmean)
            
        Returns:
            Unified performance score
        """
        w_f1, w_auc, w_gmean = weights
        
        # Handle edge cases where gmean might be 0
        if gmean == 0:
            # Use only F1 and AUC when gmean is invalid
            unified = (f1 * 0.6) + (auc * 0.4)
        else:
            # Use all three metrics
            unified = (f1 * w_f1) + (auc * w_auc) + (gmean * w_gmean)
        
        return unified
    
    @staticmethod
    def adaptive_metric_selection(f1: float, auc: float, gmean: float) -> float:
        """
        Adaptively select the most appropriate metric based on data characteristics
        
        Returns:
            Selected performance metric
        """
        # If gmean is valid and reasonable, use it as primary metric
        if gmean > 0.1:  # Threshold for valid gmean
            return gmean
        # Otherwise, prioritize F1 for imbalanced datasets
        elif f1 > 0:
            return f1
        # Fallback to AUC
        else:
            return auc


class LLaPipeAdaptiveAdvisor:
    """
    Adaptive Advisor Triggering mechanism from LLaPipe paper
    
    Key features:
    1. Uncertainty-based triggering when RL agent is uncertain
    2. Performance stagnation detection using linear regression
    3. Cooldown period to prevent excessive LLM calls
    """
    
    def __init__(self,
                 slope_threshold: float = 0.01,  # LLaPipe paper uses 0.01
                 buffer_size: int = 10,          # LLaPipe paper uses max 10 episodes
                 cooldown_period: int = 5,       # LLaPipe paper uses 5-episode cooldown
                 uncertainty_threshold: float = 0.1,
                 min_episodes_before_trigger: int = 5):  # LLaPipe paper requires 5+ episodes
        """
        Args:
            slope_threshold: Minimum learning progress required to continue RL
            buffer_size: Size of performance buffer for slope analysis  
            cooldown_period: Minimum episodes between LLM interventions
            uncertainty_threshold: Threshold for Q-value uncertainty detection
            min_episodes_before_trigger: Minimum episodes before allowing first trigger
        """
        self.slope_threshold = slope_threshold
        self.buffer_size = buffer_size  
        self.cooldown_period = cooldown_period
        self.uncertainty_threshold = uncertainty_threshold
        self.min_episodes_before_trigger = min_episodes_before_trigger
        
        # Performance tracking for stagnation detection
        self.performance_buffer = deque(maxlen=buffer_size)
        self.episode_count = 0
        self.last_llm_episode = -cooldown_period
        
        # Statistics tracking
        self.trigger_count = 0
        self.uncertainty_triggers = 0
        self.stagnation_triggers = 0
        self.successful_interventions = 0
        self.slope_history = []
        
        print(f"🎯 LLaPipe Adaptive Advisor initialized (following original paper):")
        print(f"   Slope threshold (β): {slope_threshold} (paper uses 0.01)")
        print(f"   Buffer size: {buffer_size} episodes (paper uses max 10)")
        print(f"   Cooldown period: {cooldown_period} episodes (paper uses 5)")
        print(f"   Min episodes: {min_episodes_before_trigger} (paper requires 5+)")
        print(f"   Slope formula: β = Σ(i - ī)(acc_i - acc̄) / Σ(i - ī)²")
    
    def detect_uncertainty(self, q_values: torch.Tensor, action_probs: torch.Tensor = None) -> Tuple[bool, float]:
        """
        Detect uncertainty in RL agent's decision making
        
        Args:
            q_values: Q-values from the critic network
            action_probs: Action probabilities (optional)
            
        Returns:
            Tuple of (is_uncertain, uncertainty_score)
        """
        if q_values is None or len(q_values) == 0:
            return False, 0.0
        
        # Q-value variance (uncertainty in value estimates)
        q_variance = torch.var(q_values).item() if len(q_values) > 1 else 0.0
        
        # Action probability entropy (if available)
        entropy_uncertainty = 0.0
        if action_probs is not None and len(action_probs) > 1:
            # Calculate entropy: -sum(p * log(p))
            probs = torch.softmax(action_probs, dim=-1)
            log_probs = torch.log(probs + 1e-8)
            entropy_uncertainty = -torch.sum(probs * log_probs).item()
            # Normalize entropy to [0,1] range
            max_entropy = np.log(len(probs))
            entropy_uncertainty = entropy_uncertainty / max_entropy if max_entropy > 0 else 0.0
        
        # Combine uncertainties
        combined_uncertainty = max(q_variance, entropy_uncertainty)
        is_uncertain = combined_uncertainty > self.uncertainty_threshold
        
        return is_uncertain, combined_uncertainty
    
    def detect_stagnation(self, current_performance: float) -> Tuple[bool, Dict]:
        """
        Detect learning stagnation using linear regression on recent performance
        
        Args:
            current_performance: Current episode performance metric
            
        Returns:
            Tuple of (is_stagnant, analysis_info)
        """
        # Add current performance to buffer
        self.performance_buffer.append({
            'performance': current_performance,
            'episode': self.episode_count,
            'timestamp': time.time()
        })
        
        analysis_info = {
            'slope': None,
            'buffer_size': len(self.performance_buffer),
            'slope_threshold': self.slope_threshold,
            'sufficient_data': False
        }
        
        # Need at least 3 data points for meaningful regression
        if len(self.performance_buffer) < 3:
            return False, analysis_info
        
        analysis_info['sufficient_data'] = True

        # Calculate slope using LLaPipe paper's formula
        try:
            # Extract episodes and performances
            episodes = np.array([p['episode'] for p in self.performance_buffer])
            performances = np.array([p['performance'] for p in self.performance_buffer])

            # Calculate means
            i_mean = np.mean(episodes)
            acc_mean = np.mean(performances)

            # LLaPipe formula: β = Σ(i - ī)(acc_i - acc̄) / Σ(i - ī)²
            numerator = np.sum((episodes - i_mean) * (performances - acc_mean))
            denominator = np.sum((episodes - i_mean) ** 2)

            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator

            self.slope_history.append(slope)
            analysis_info['slope'] = slope
            analysis_info['formula'] = 'LLaPipe_manual_regression'

            # Check if slope indicates stagnation (LLaPipe paper uses β < threshold)
            is_stagnant = slope < self.slope_threshold

            return is_stagnant, analysis_info
            
        except Exception as e:
            print(f"⚠️ Regression analysis failed: {e}")
            return False, analysis_info
    
    def should_trigger_llm(self, 
                          current_performance: float,
                          q_values: torch.Tensor = None,
                          action_probs: torch.Tensor = None) -> Tuple[bool, Dict]:
        """
        Main decision function: should we trigger LLM intervention?
        
        Args:
            current_performance: Current episode performance
            q_values: Q-values from critic (for uncertainty detection)
            action_probs: Action probabilities (for uncertainty detection)
            
        Returns:
            Tuple of (should_trigger, decision_info)
        """
        self.episode_count += 1
        
        decision_info = {
            'episode': self.episode_count,
            'trigger_reason': None,
            'uncertainty_detected': False,
            'stagnation_detected': False,
            'uncertainty_score': 0.0,
            'learning_slope': None,
            'cooldown_active': False,
            'sufficient_episodes': False
        }
        
        # Check minimum episodes requirement
        if self.episode_count < self.min_episodes_before_trigger:
            decision_info['trigger_reason'] = 'insufficient_episodes'
            return False, decision_info
        
        decision_info['sufficient_episodes'] = True
        
        # Check cooldown period
        episodes_since_last = self.episode_count - self.last_llm_episode
        if episodes_since_last < self.cooldown_period:
            decision_info['trigger_reason'] = 'cooldown_active'
            decision_info['cooldown_active'] = True
            decision_info['episodes_since_last'] = episodes_since_last
            return False, decision_info
        
        # Check for uncertainty in RL agent's decisions
        uncertainty_detected = False
        uncertainty_score = 0.0
        
        if q_values is not None:
            uncertainty_detected, uncertainty_score = self.detect_uncertainty(q_values, action_probs)
            decision_info['uncertainty_detected'] = uncertainty_detected
            decision_info['uncertainty_score'] = uncertainty_score
        
        # Check for learning stagnation
        stagnation_detected, stagnation_info = self.detect_stagnation(current_performance)
        decision_info['stagnation_detected'] = stagnation_detected
        decision_info['learning_slope'] = stagnation_info.get('slope')
        decision_info.update(stagnation_info)
        
        # Decision logic: trigger if either uncertainty or stagnation detected
        should_trigger = uncertainty_detected or stagnation_detected
        
        if should_trigger:
            self.trigger_count += 1
            self.last_llm_episode = self.episode_count
            
            if uncertainty_detected and stagnation_detected:
                decision_info['trigger_reason'] = 'uncertainty_and_stagnation'
                self.uncertainty_triggers += 1
                self.stagnation_triggers += 1
            elif uncertainty_detected:
                decision_info['trigger_reason'] = 'uncertainty'
                self.uncertainty_triggers += 1
            else:
                decision_info['trigger_reason'] = 'stagnation'
                self.stagnation_triggers += 1
            
            print(f"🚨 LLM Intervention Triggered!")
            print(f"   Reason: {decision_info['trigger_reason']}")
            print(f"   Episode: {self.episode_count}")
            if uncertainty_detected:
                print(f"   Uncertainty score: {uncertainty_score:.4f}")
            if stagnation_detected and decision_info['learning_slope'] is not None:
                print(f"   Learning slope: {decision_info['learning_slope']:.6f}")
        else:
            decision_info['trigger_reason'] = 'no_intervention_needed'
        
        return should_trigger, decision_info
    
    def update_intervention_result(self, success: bool, performance_gain: float = 0.0):
        """Update statistics based on intervention outcome"""
        if success:
            self.successful_interventions += 1
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        total_episodes = max(self.episode_count, 1)
        
        stats = {
            'total_episodes': self.episode_count,
            'total_triggers': self.trigger_count,
            'uncertainty_triggers': self.uncertainty_triggers,
            'stagnation_triggers': self.stagnation_triggers,
            'successful_interventions': self.successful_interventions,
            'trigger_rate': self.trigger_count / total_episodes,
            'success_rate': self.successful_interventions / max(self.trigger_count, 1),
            'avg_episodes_between_triggers': total_episodes / max(self.trigger_count, 1),
            'slope_threshold': self.slope_threshold,
            'uncertainty_threshold': self.uncertainty_threshold,
            'cooldown_period': self.cooldown_period
        }
        
        # Add slope statistics if available
        if self.slope_history:
            stats.update({
                'avg_slope': np.mean(self.slope_history),
                'slope_std': np.std(self.slope_history),
                'min_slope': np.min(self.slope_history),
                'max_slope': np.max(self.slope_history),
                'recent_slope': self.slope_history[-1]
            })
        
        return stats
    
    def print_statistics(self):
        """Print comprehensive statistics"""
        stats = self.get_statistics()
        
        print(f"\n📊 LLaPipe Adaptive Advisor Statistics:")
        print(f"   Total episodes: {stats['total_episodes']}")
        print(f"   LLM interventions: {stats['total_triggers']} ({stats['trigger_rate']:.1%})")
        print(f"   - Uncertainty triggers: {stats['uncertainty_triggers']}")
        print(f"   - Stagnation triggers: {stats['stagnation_triggers']}")
        print(f"   Successful interventions: {stats['successful_interventions']} ({stats['success_rate']:.1%})")
        print(f"   Avg episodes between triggers: {stats['avg_episodes_between_triggers']:.1f}")
        
        if 'avg_slope' in stats:
            print(f"   Average learning slope: {stats['avg_slope']:.6f}")
            print(f"   Recent slope: {stats['recent_slope']:.6f}")