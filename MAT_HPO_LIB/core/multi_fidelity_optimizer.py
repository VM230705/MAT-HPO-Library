"""
Multi-Fidelity MAT-HPO Integration

Extends the MAT-HPO framework to support multi-fidelity optimization strategies.
Provides seamless integration between progressive resource allocation and 
multi-agent reinforcement learning for hyperparameter optimization.
"""

import os
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from .multi_agent_optimizer import MAT_HPO_Optimizer
from .base_environment import BaseEnvironment  
from .hyperparameter_space import HyperparameterSpace
from ..utils.config import OptimizationConfig
from ..utils.logger import HPOLogger

@dataclass
class FidelityConfig:
    """Configuration for multi-fidelity optimization"""
    min_fidelity: int = 5      # Minimum training epochs
    max_fidelity: int = 50     # Maximum training epochs
    fidelity_levels: List[int] = None  # Custom fidelity levels
    promotion_factor: float = 0.5      # Fraction to promote to next level
    elimination_factor: float = 0.3    # Fraction to eliminate at each level
    early_stop_patience: int = 3       # Epochs to wait for improvement
    performance_threshold: float = 0.1 # Minimum performance to continue

    def __post_init__(self):
        if self.fidelity_levels is None:
            self.fidelity_levels = [5, 15, 30, 50]

class MultiFidelityEnvironment(BaseEnvironment):
    """Wrapper environment that supports multi-fidelity evaluation"""
    
    def __init__(self, base_environment: BaseEnvironment, fidelity_config: FidelityConfig):
        super().__init__(
            name=f"MultiFidelity_{base_environment.name}",
            validation_split=base_environment.validation_split
        )
        self.base_environment = base_environment
        self.fidelity_config = fidelity_config
        self.candidate_fidelities = {}  # Track current fidelity for each candidate
        self.candidate_histories = {}   # Track evaluation history
        
    def load_data(self):
        """Delegate data loading to base environment"""
        return self.base_environment.load_data()
    
    def create_model(self, hyperparams: Dict[str, Any]):
        """Delegate model creation to base environment"""
        return self.base_environment.create_model(hyperparams)
    
    def train_evaluate_at_fidelity(self, model, hyperparams: Dict[str, Any], 
                                 fidelity: int, candidate_id: str = None) -> Dict[str, float]:
        """
        Train and evaluate at specific fidelity level
        
        Args:
            model: Model to train (can be None if created internally)
            hyperparams: Hyperparameter configuration
            fidelity: Number of training epochs/iterations
            candidate_id: Unique identifier for tracking
            
        Returns:
            Dictionary with metrics including fidelity info
        """
        # Add fidelity information to hyperparams
        extended_hyperparams = hyperparams.copy()
        extended_hyperparams['_fidelity'] = fidelity
        extended_hyperparams['_candidate_id'] = candidate_id
        
        # Check if base environment supports fidelity
        if hasattr(self.base_environment, 'train_evaluate_with_fidelity'):
            metrics = self.base_environment.train_evaluate_with_fidelity(
                model, extended_hyperparams, fidelity
            )
        elif hasattr(self.base_environment, 'train_evaluate_with_early_stop'):
            metrics = self.base_environment.train_evaluate_with_early_stop(
                model, extended_hyperparams, fidelity, candidate_id or "unknown"
            )
        else:
            # Fallback to regular training (may not respect fidelity)
            metrics = self.base_environment.train_evaluate(model, extended_hyperparams)
        
        # Ensure fidelity information is included in results
        metrics['fidelity'] = fidelity
        metrics['candidate_id'] = candidate_id
        
        return metrics
    
    def train_evaluate(self, model, hyperparams: Dict[str, Any]) -> Dict[str, float]:
        """Standard interface - delegates to base environment"""
        return self.base_environment.train_evaluate(model, hyperparams)
    
    def compute_reward(self, metrics: Dict[str, float]) -> float:
        """Enhanced reward computation considering fidelity"""
        base_reward = self.base_environment.compute_reward(metrics)
        
        # Fidelity-adjusted reward
        fidelity = metrics.get('fidelity', self.fidelity_config.max_fidelity)
        fidelity_factor = fidelity / self.fidelity_config.max_fidelity
        
        # Adjust reward based on fidelity level
        # Lower fidelity results get slight penalty to encourage full training
        if fidelity < self.fidelity_config.max_fidelity:
            base_reward *= (0.9 + 0.1 * fidelity_factor)  # 10% max penalty
        
        return base_reward

class MultiFidelityMAT_HPO:
    """
    Multi-Fidelity MAT-HPO Optimizer
    
    Integrates progressive resource allocation with multi-agent reinforcement learning
    for efficient hyperparameter optimization.
    """
    
    def __init__(self, 
                 environment: BaseEnvironment,
                 hyperparameter_space: HyperparameterSpace,
                 config: OptimizationConfig,
                 fidelity_config: Optional[FidelityConfig] = None):
        """
        Initialize multi-fidelity optimizer
        
        Args:
            environment: Base environment for evaluation
            hyperparameter_space: Hyperparameter space definition
            config: Optimization configuration
            fidelity_config: Multi-fidelity specific configuration
        """
        self.fidelity_config = fidelity_config or FidelityConfig()
        
        # Wrap environment with multi-fidelity support
        self.mf_environment = MultiFidelityEnvironment(environment, self.fidelity_config)
        
        # Initialize base MAT-HPO optimizer
        self.mat_hpo = MAT_HPO_Optimizer(
            environment=self.mf_environment,
            hyperparameter_space=hyperparameter_space,
            config=config
        )
        
        # Multi-fidelity state management
        self.candidates = {}  # candidate_id -> hyperparams
        self.fidelity_results = {}  # candidate_id -> {fidelity: results}
        self.active_candidates = {}  # fidelity_level -> [candidate_ids]
        
        # Initialize fidelity tracking
        for fidelity in self.fidelity_config.fidelity_levels:
            self.active_candidates[fidelity] = []
        
        # Logging
        self.logger = logging.getLogger(f"MultiFidelityMAT_HPO")
        self.optimization_history = []
    
    def optimize(self, max_candidates: int = 20) -> Dict[str, Any]:
        """
        Run multi-fidelity optimization
        
        Args:
            max_candidates: Maximum number of candidates to evaluate
            
        Returns:
            Optimization results with best hyperparameters and performance
        """
        self.logger.info(f"Starting multi-fidelity optimization with {max_candidates} candidates")
        
        optimization_start = time.time()
        total_evaluations = 0
        
        # Phase 1: Generate initial candidate pool at lowest fidelity
        min_fidelity = self.fidelity_config.fidelity_levels[0]
        initial_candidates = self._generate_initial_candidates(max_candidates, min_fidelity)
        total_evaluations += len(initial_candidates)
        
        self.logger.info(f"Generated {len(initial_candidates)} initial candidates at fidelity {min_fidelity}")
        
        # Phase 2: Progressive evaluation through fidelity levels
        for fidelity_idx, fidelity in enumerate(self.fidelity_config.fidelity_levels):
            
            self.logger.info(f"Phase {fidelity_idx + 2}: Evaluating at fidelity {fidelity}")
            
            # Get candidates for this fidelity level
            if fidelity_idx == 0:
                # First fidelity level - use all initial candidates
                candidates_to_evaluate = initial_candidates
            else:
                # Higher fidelity levels - use promoted candidates
                candidates_to_evaluate = self._select_candidates_for_fidelity(fidelity)
            
            if not candidates_to_evaluate:
                self.logger.info(f"No candidates to evaluate at fidelity {fidelity}")
                continue
            
            # Evaluate candidates at current fidelity
            fidelity_results = self._evaluate_candidates_at_fidelity(
                candidates_to_evaluate, fidelity
            )
            total_evaluations += len(fidelity_results)
            
            # Record results
            for candidate_id, result in fidelity_results.items():
                if candidate_id not in self.fidelity_results:
                    self.fidelity_results[candidate_id] = {}
                self.fidelity_results[candidate_id][fidelity] = result
            
            # Update active candidates for next fidelity level
            if fidelity_idx < len(self.fidelity_config.fidelity_levels) - 1:
                promoted_candidates = self._promote_candidates(fidelity, fidelity_results)
                next_fidelity = self.fidelity_config.fidelity_levels[fidelity_idx + 1]
                self.active_candidates[next_fidelity] = promoted_candidates
                
                self.logger.info(f"Promoted {len(promoted_candidates)} candidates to fidelity {next_fidelity}")
        
        optimization_time = time.time() - optimization_start
        
        # Phase 3: Analyze results and extract best configuration
        best_result = self._analyze_optimization_results()
        
        # Compile final results
        results = {
            'best_hyperparameters': best_result['hyperparams'],
            'best_performance': best_result['performance'],
            'best_fidelity': best_result['fidelity'],
            'optimization_time': optimization_time,
            'total_evaluations': total_evaluations,
            'fidelity_breakdown': self._get_fidelity_breakdown(),
            'efficiency_gain': self._calculate_efficiency_gain(total_evaluations),
            'full_results': self.fidelity_results
        }
        
        self.logger.info(f"Optimization completed in {optimization_time:.2f}s with {total_evaluations} evaluations")
        self.logger.info(f"Best performance: {best_result['performance']:.4f} at fidelity {best_result['fidelity']}")
        
        return results
    
    def _generate_initial_candidates(self, num_candidates: int, fidelity: int) -> List[Dict[str, Any]]:
        """Generate initial candidate pool using MAT-HPO sampling"""
        candidates = []
        
        for i in range(num_candidates):
            # Sample hyperparameters using MAT-HPO
            candidate_params = self.mat_hpo.hyperparameter_space.sample()
            
            candidate_id = f"candidate_{i}"
            self.candidates[candidate_id] = candidate_params
            
            candidates.append({
                'id': candidate_id,
                'hyperparams': candidate_params
            })
        
        return candidates
    
    def _evaluate_candidates_at_fidelity(self, candidates: List[Dict[str, Any]], 
                                       fidelity: int) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple candidates at specified fidelity"""
        results = {}
        
        for candidate in candidates:
            candidate_id = candidate['id']
            hyperparams = candidate['hyperparams']
            
            try:
                # Evaluate at specified fidelity
                metrics = self.mf_environment.train_evaluate_at_fidelity(
                    model=None,  # Let environment handle model creation
                    hyperparams=hyperparams,
                    fidelity=fidelity,
                    candidate_id=candidate_id
                )
                
                results[candidate_id] = metrics
                
                # Log progress
                performance = metrics.get('val_accuracy', metrics.get('f1', 0))
                self.logger.debug(f"Candidate {candidate_id} at fidelity {fidelity}: {performance:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate candidate {candidate_id}: {str(e)}")
                # Assign poor performance for failed evaluations
                results[candidate_id] = {
                    'val_accuracy': 0.001,
                    'f1': 0.001,
                    'fidelity': fidelity,
                    'failed': True,
                    'error': str(e)
                }
        
        return results
    
    def _select_candidates_for_fidelity(self, fidelity: int) -> List[Dict[str, Any]]:
        """Select candidates to evaluate at given fidelity level"""
        # Find previous fidelity level
        prev_fidelity = None
        for f in reversed(self.fidelity_config.fidelity_levels):
            if f < fidelity:
                prev_fidelity = f
                break
        
        if prev_fidelity is None:
            return []
        
        # Get candidates from active list
        candidate_ids = self.active_candidates.get(fidelity, [])
        
        candidates = []
        for candidate_id in candidate_ids:
            if candidate_id in self.candidates:
                candidates.append({
                    'id': candidate_id,
                    'hyperparams': self.candidates[candidate_id]
                })
        
        return candidates
    
    def _promote_candidates(self, current_fidelity: int, 
                          results: Dict[str, Dict[str, float]]) -> List[str]:
        """Select candidates for promotion to next fidelity level"""
        if not results:
            return []
        
        # Sort candidates by performance
        performance_key = 'val_accuracy'  # Primary metric
        candidate_performance = []
        
        for candidate_id, metrics in results.items():
            performance = metrics.get(performance_key, metrics.get('f1', 0))
            candidate_performance.append((candidate_id, performance))
        
        # Sort by performance (descending)
        candidate_performance.sort(key=lambda x: x[1], reverse=True)
        
        # Select top candidates for promotion
        num_to_promote = max(1, int(len(candidate_performance) * self.fidelity_config.promotion_factor))
        promoted = [candidate_id for candidate_id, _ in candidate_performance[:num_to_promote]]
        
        return promoted
    
    def _analyze_optimization_results(self) -> Dict[str, Any]:
        """Find the best performing candidate across all fidelities"""
        best_candidate = None
        best_performance = -float('inf')
        best_fidelity = None
        
        for candidate_id, fidelity_results in self.fidelity_results.items():
            for fidelity, metrics in fidelity_results.items():
                # Use validation accuracy as primary metric
                performance = metrics.get('val_accuracy', metrics.get('f1', 0))
                
                if performance > best_performance:
                    best_performance = performance
                    best_candidate = candidate_id
                    best_fidelity = fidelity
        
        if best_candidate is None:
            return {
                'hyperparams': {},
                'performance': 0.0,
                'fidelity': 0,
                'candidate_id': None
            }
        
        return {
            'hyperparams': self.candidates[best_candidate],
            'performance': best_performance,
            'fidelity': best_fidelity,
            'candidate_id': best_candidate
        }
    
    def _get_fidelity_breakdown(self) -> Dict[int, int]:
        """Get count of evaluations per fidelity level"""
        breakdown = {}
        
        for candidate_results in self.fidelity_results.values():
            for fidelity in candidate_results.keys():
                breakdown[fidelity] = breakdown.get(fidelity, 0) + 1
        
        return breakdown
    
    def _calculate_efficiency_gain(self, total_evaluations: int) -> Dict[str, float]:
        """Calculate efficiency gains compared to full-fidelity evaluation"""
        max_fidelity = self.fidelity_config.max_fidelity
        
        # Calculate total epochs used
        total_epochs_used = 0
        for candidate_results in self.fidelity_results.values():
            for fidelity, metrics in candidate_results.items():
                total_epochs_used += fidelity
        
        # Calculate epochs if all evaluations were done at max fidelity
        total_epochs_full_fidelity = total_evaluations * max_fidelity
        
        # Calculate savings
        epochs_saved = total_epochs_full_fidelity - total_epochs_used
        efficiency_percentage = (epochs_saved / total_epochs_full_fidelity) * 100
        
        return {
            'total_epochs_used': total_epochs_used,
            'total_epochs_full_fidelity': total_epochs_full_fidelity,
            'epochs_saved': epochs_saved,
            'efficiency_percentage': efficiency_percentage
        }