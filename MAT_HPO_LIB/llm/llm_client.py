"""
LLM Client for MAT_HPO_LIB - Ollama Integration with Adaptive Methods
Adapted from MAT_HPO_LLM implementation for the reusable library
"""

import requests
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import time

from .adaptive_advisor import LLaPipeAdaptiveAdvisor, PerformanceMetricCalculator
from .dataset_info_reader import get_enhanced_dataset_info, get_dataset_recommendations
from .conversation_logger import LLMConversationLogger


class OllamaLLMClient:
    """
    LLM Client for Ollama integration with MAT_HPO_LIB

    Provides unified hyperparameter generation for all 3 agents using
    Large Language Models through the Ollama API.
    """

    def __init__(self, model_name="llama3.2:3b", base_url="http://localhost:11434", dataset_info_csv_path="./Datasets_info.csv"):
        """
        Initialize Ollama LLM Client

        Args:
            model_name: Ollama model name (default: llama3.2:3b)
            base_url: Ollama service URL (default: http://localhost:11434)
            dataset_info_csv_path: Path to dataset information CSV file
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.dataset_info_csv_path = dataset_info_csv_path

        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            response = requests.post(self.api_url, json={
                "model": self.model_name,
                "prompt": "Hello",
                "stream": False
            }, timeout=10)
            if response.status_code == 200:
                print(f"✅ Connected to Ollama {self.model_name}")
            else:
                print(f"⚠️ Ollama connection issue: {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to connect to Ollama: {e}")
            print("Make sure Ollama is running: 'ollama serve' and model is pulled: 'ollama pull llama3.2:3b'")
    
    def generate_hyperparameters(self, 
                               hyperparameter_space,
                               dataset_info: Dict, 
                               training_history: List[Dict]) -> Dict[str, Any]:
        """
        Generate hyperparameters for all agents using LLM
        
        Args:
            hyperparameter_space: HyperparameterSpace object defining the search space
            dataset_info: Dataset characteristics
            training_history: Previous training results
            
        Returns:
            Dictionary of hyperparameter names to values
        """
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Create unified prompt
                prompt = self._create_unified_prompt(hyperparameter_space, dataset_info, training_history, 
                                                   previous_error=last_error if attempt > 0 else None)
                
                response = requests.post(self.api_url, json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                }, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    try:
                        parsed_params = self._parse_hyperparameters(
                            result["response"], hyperparameter_space)
                        
                        # Log successful conversation if logger available
                        if hasattr(self, 'conversation_logger') and self.conversation_logger:
                            self.conversation_logger.log_llm_conversation(
                                step=getattr(self, 'current_step', 0),
                                attempt=attempt + 1,
                                prompt=prompt,
                                response=result["response"],
                                parse_success=True,
                                parsed_params=parsed_params,
                                dataset_info=dataset_info,
                                training_history=training_history
                            )
                        
                        return parsed_params
                        
                    except Exception as parse_error:
                        last_error = f"Parse error: {str(parse_error)}. Response was: {result['response']}"
                        
                        # Log failed conversation
                        if hasattr(self, 'conversation_logger') and self.conversation_logger:
                            self.conversation_logger.log_llm_conversation(
                                step=getattr(self, 'current_step', 0),
                                attempt=attempt + 1,
                                prompt=prompt,
                                response=result["response"],
                                parse_success=False,
                                error_message=last_error,
                                dataset_info=dataset_info,
                                training_history=training_history
                            )
                        
                        print(f"🔄 Attempt {attempt + 1}/3 failed: {last_error}")
                        continue
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    print(f"🔄 Attempt {attempt + 1}/3 failed: {last_error}")
                    continue
                    
            except Exception as e:
                last_error = f"Request error: {str(e)}"
                print(f"🔄 Attempt {attempt + 1}/3 failed: {last_error}")
                continue
        
        # All attempts failed - use fallback
        print("❌ ALL LLM ATTEMPTS FAILED - Using fallback hyperparameters")
        fallback_params = self._fallback_hyperparameters(hyperparameter_space)
        
        # Log failure
        if hasattr(self, 'conversation_logger') and self.conversation_logger:
            self.conversation_logger.log_failure_summary(
                step=getattr(self, 'current_step', 0),
                all_attempts=[{"error_message": last_error}],
                fallback_params=fallback_params
            )
        
        return fallback_params
    
    def _create_unified_prompt(self, hyperparameter_space, dataset_info: Dict, 
                             training_history: List[Dict], previous_error: str = None) -> str:
        """Create unified prompt for hyperparameter generation"""
        
        # Get dataset name for enhanced info
        dataset_name = dataset_info.get('name', 'Unknown')
        # Create DatasetInfoReader with custom CSV path
        from .dataset_info_reader import DatasetInfoReader
        reader = DatasetInfoReader(self.dataset_info_csv_path)
        enhanced_info = reader.get_dataset_stats(dataset_name)
        recommendations = reader.get_optimization_recommendations(dataset_name)
        
        # Dataset context
        if enhanced_info:
            dataset_context = f"""
Dataset: {enhanced_info['name']} ({'Univariate' if enhanced_info['univariate'] else 'Multivariate'} Time Series)
Total Samples: {enhanced_info['total_samples']:,} ({enhanced_info['dataset_size_category']})
Features: {enhanced_info['num_features']}
Sequence Length: {enhanced_info['sequence_length']}
Classes: {enhanced_info['num_classes']}
Class Imbalance: {enhanced_info['max_imbalance_ratio']:.2f} ({enhanced_info['imbalance_severity']})
Problem Complexity: {enhanced_info['complexity_level']}
"""
        else:
            # Fallback context
            dataset_context = f"""
Dataset: {dataset_name}
Training Samples: {dataset_info.get('total_samples', 'Unknown')}
Features: {dataset_info.get('num_features', 'Unknown')}
Classes: {dataset_info.get('num_classes', 'Unknown')}
"""
        
        # Training history context
        history_context = ""
        if training_history:
            recent = training_history[-3:]  # Last 3 results
            history_context = "Recent Training Results:\n"
            for i, result in enumerate(recent):
                metrics = result.get('metrics', {})
                f1 = metrics.get('f1', result.get('f1', 0))
                auc = metrics.get('auc', result.get('auc', 0))
                gmean = metrics.get('gmean', result.get('gmean', 0))
                history_context += f"Step {len(training_history)-len(recent)+i+1}: F1={f1:.3f}, AUC={auc:.3f}, GMean={gmean:.3f}\n"
        
        # Hyperparameter space description
        param_descriptions = []
        if hasattr(hyperparameter_space, 'parameters'):
            for param_name, param_info in hyperparameter_space.parameters.items():
                agent_id = param_info.get('agent', 0)
                param_type = param_info.get('type', 'continuous')
                if param_type == 'continuous':
                    min_val, max_val = param_info['bounds']
                    param_descriptions.append(f"Agent {agent_id}: {param_name} ∈ [{min_val}, {max_val}]")
                elif param_type == 'discrete':
                    choices = param_info['choices']
                    param_descriptions.append(f"Agent {agent_id}: {param_name} ∈ {choices}")
        
        # Error feedback if this is a retry
        error_feedback = ""
        if previous_error:
            error_feedback = f"""
❌ PREVIOUS ATTEMPT FAILED: {previous_error}

CRITICAL: You must provide hyperparameters in JSON format as specified below.
"""

        # Smart guidance
        if enhanced_info and recommendations:
            guidance = self._get_smart_guidance(enhanced_info, recommendations)
        else:
            guidance = "Use balanced hyperparameters appropriate for the dataset characteristics."

        # Build specific parameter list for JSON output
        specific_params = []
        if hasattr(hyperparameter_space, 'parameters'):
            for param_name in hyperparameter_space.parameters.keys():
                specific_params.append(f'  "{param_name}": <value>')
        
        specific_params_str = ",\n".join(specific_params) if specific_params else '  "parameter_name": <value>'
        
        prompt = f"""
You are an expert in hyperparameter optimization for machine learning.

{dataset_context}
{history_context}

Hyperparameter Search Space:
{chr(10).join(param_descriptions)}

{guidance}
{error_feedback}

Based on the dataset characteristics and training history, provide optimal hyperparameters.

OUTPUT FORMAT (JSON):
{{
{specific_params_str}
}}

Provide ONLY the JSON object with the EXACT hyperparameter names listed above and their values. No explanations.
"""
        
        return prompt
    
    def _get_smart_guidance(self, enhanced_info: Dict, recommendations: Dict) -> str:
        """Generate smart guidance based on dataset characteristics"""
        guidance_parts = []
        
        # Dataset size guidance
        size_category = enhanced_info['dataset_size_category']
        if size_category == 'VERY_SMALL':
            guidance_parts.append("VERY SMALL DATASET: Use minimal model capacity and aggressive regularization.")
        elif size_category == 'SMALL':
            guidance_parts.append("SMALL DATASET: Use conservative parameters to prevent overfitting.")
        elif size_category == 'MEDIUM':
            guidance_parts.append("MEDIUM DATASET: Balance model capacity with generalization.")
        else:
            guidance_parts.append("LARGE DATASET: Can use larger models and complex architectures.")
        
        # Imbalance guidance
        imbalance = enhanced_info['imbalance_severity']
        if imbalance == 'SEVERE_IMBALANCE':
            guidance_parts.append("SEVERE IMBALANCE: Use aggressive class weighting and careful validation.")
        elif imbalance == 'MODERATE_IMBALANCE':
            guidance_parts.append("MODERATE IMBALANCE: Apply balanced class weighting strategy.")
        
        return " ".join(guidance_parts)
    
    def _parse_hyperparameters(self, llm_response: str, hyperparameter_space) -> Dict[str, Any]:
        """Parse LLM response into hyperparameter dictionary"""
        try:
            # Try to extract JSON from response
            import re
            import json
            
            # Clean response - remove markdown code blocks
            cleaned_response = llm_response
            if '```' in cleaned_response:
                # Extract content between ```json and ``` or just between ```
                json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned_response, re.DOTALL)
                if json_match:
                    cleaned_response = json_match.group(1).strip()
            
            # Try to find JSON object with proper bracket matching
            start_idx = cleaned_response.find('{')
            if start_idx != -1:
                bracket_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(cleaned_response)):
                    if cleaned_response[i] == '{':
                        bracket_count += 1
                    elif cleaned_response[i] == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i
                            break
                
                json_str = cleaned_response[start_idx:end_idx + 1]
                parsed = json.loads(json_str)
                
                # Validate and bound parameters
                bounded_params = {}
                if hyperparameter_space is not None and hasattr(hyperparameter_space, 'parameters'):
                    for param_name, param_info in hyperparameter_space.parameters.items():
                        if param_name in parsed:
                            value = parsed[param_name]
                            param_type = param_info.get('type', 'continuous')
                            
                            if param_type == 'continuous':
                                min_val, max_val = param_info['bounds']
                                bounded_params[param_name] = max(min_val, min(max_val, float(value)))
                            elif param_type == 'discrete':
                                choices = param_info['choices']
                                if value in choices:
                                    bounded_params[param_name] = value
                                else:
                                    # Choose closest or default
                                    bounded_params[param_name] = choices[0]
                            else:
                                bounded_params[param_name] = value
                else:
                    # If no hyperparameter space provided, return parsed params as-is
                    bounded_params = parsed
                
                return bounded_params
            
            raise ValueError("No valid JSON found in response")
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"LLM response: {llm_response}")
            return self._fallback_hyperparameters(hyperparameter_space)
    
    def _fallback_hyperparameters(self, hyperparameter_space) -> Dict[str, Any]:
        """Generate fallback hyperparameters when LLM fails"""
        fallback_params = {}
        
        if hasattr(hyperparameter_space, 'parameters'):
            for param_name, param_info in hyperparameter_space.parameters.items():
                param_type = param_info.get('type', 'continuous')
                
                if param_type == 'continuous':
                    min_val, max_val = param_info['bounds']
                    # Use middle value as fallback
                    fallback_params[param_name] = (min_val + max_val) / 2
                elif param_type == 'discrete':
                    choices = param_info['choices']
                    # Use first choice as fallback
                    fallback_params[param_name] = choices[0]
        
        return fallback_params


class LLMHyperparameterMixer:
    """
    Mixes RL and LLM hyperparameter suggestions with adaptive triggering
    
    Supports both fixed alpha mixing and adaptive triggering based on the
    LLaPipe paper methodology.
    """
    
    def __init__(self, alpha: float = 0.3, dataset_name: str = "Unknown", output_dir: str = None,
                 use_adaptive_trigger: bool = False, slope_threshold: float = 0.01):
        """
        Args:
            alpha: Probability of using LLM suggestions (0.0 = always RL, 1.0 = always LLM)
                   Ignored if use_adaptive_trigger=True
            dataset_name: Name of the dataset for logging
            output_dir: Directory to store logs for this run
            use_adaptive_trigger: Whether to use adaptive triggering instead of fixed alpha
            slope_threshold: Slope threshold for adaptive triggering
        """
        self.alpha = alpha
        self.dataset_name = dataset_name
        self.llm_client = OllamaLLMClient()
        self.training_history = []
        self.rl_decision_count = 0
        self.llm_decision_count = 0
        
        # Adaptive triggering setup
        self.use_adaptive_trigger = use_adaptive_trigger
        if self.use_adaptive_trigger:
            self.adaptive_advisor = LLaPipeAdaptiveAdvisor(
                slope_threshold=slope_threshold,
                buffer_size=10,
                cooldown_period=5,
                uncertainty_threshold=0.1,
                min_episodes_before_trigger=5
            )
            print(f"🎯 LLaPipe Adaptive triggering enabled for {dataset_name} (slope_threshold={slope_threshold})")
        else:
            self.adaptive_advisor = None
            print(f"📊 Fixed alpha strategy enabled for {dataset_name} (alpha={alpha})")
        
        # Random generator for LLM decisions (fixed alpha mode only)
        self.decision_rng = np.random.RandomState()
        
        # Initialize conversation logger
        mode_str = f"adaptive_{slope_threshold}" if use_adaptive_trigger else f"alpha_{alpha}"
        self.conversation_logger = LLMConversationLogger(dataset_name, mode_str, output_dir=output_dir)
        
        # Set logger in LLM client
        self.llm_client.conversation_logger = self.conversation_logger
    
    def should_use_llm(self, 
                      current_performance: Optional[Dict] = None,
                      q_values: Optional[torch.Tensor] = None,
                      action_probs: Optional[torch.Tensor] = None) -> Tuple[bool, Dict]:
        """
        Decide whether to use LLM suggestions
        
        Args:
            current_performance: Dict with 'f1', 'auc', 'gmean' keys for adaptive mode
            q_values: Q-values for uncertainty detection (adaptive mode)
            action_probs: Action probabilities for uncertainty detection (adaptive mode)
            
        Returns:
            Tuple of (should_use_llm, decision_info)
        """
        if self.use_adaptive_trigger and self.adaptive_advisor is not None:
            # Use adaptive triggering
            if current_performance is None:
                return False, {"method": "adaptive", "reason": "no_performance_data"}
            
            # Calculate unified performance metric
            unified_metric = PerformanceMetricCalculator.adaptive_metric_selection(
                current_performance.get('f1', 0),
                current_performance.get('auc', 0),
                current_performance.get('gmean', 0)
            )
            
            # Check if LLM should be triggered
            should_trigger, decision_info = self.adaptive_advisor.should_trigger_llm(
                current_performance=unified_metric,
                q_values=q_values,
                action_probs=action_probs
            )
            decision_info['method'] = 'adaptive'
            decision_info['unified_metric'] = unified_metric
            
            return should_trigger, decision_info
        else:
            # Use fixed alpha probability
            use_llm = self.decision_rng.random() < self.alpha
            decision_info = {
                'method': 'fixed_alpha',
                'alpha': self.alpha,
                'random_draw': use_llm
            }
            return use_llm, decision_info
    
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
            current_performance: Recent performance for adaptive triggering
            q_values: Q-values for uncertainty detection
            
        Returns:
            Tuple of (final_hyperparams, decision_source)
        """
        # Set current step for logging
        self.llm_client.current_step = step
        
        # Decide whether to use LLM
        use_llm, decision_info = self.should_use_llm(current_performance, q_values)
        
        if use_llm:
            self.llm_decision_count += 1
            decision_source = "LLM"
            
            try:
                llm_hyperparams = self.llm_client.generate_hyperparameters(
                    hyperparameter_space=hyperparameter_space,
                    dataset_info=dataset_info,
                    training_history=self.training_history
                )
                
                # Log decision info
                if self.use_adaptive_trigger:
                    reason = decision_info.get('trigger_reason', 'unknown')
                    slope = decision_info.get('learning_slope', 0)
                    if slope is not None:
                        print(f"🤖 LLM triggered by adaptive advisor (reason: {reason}, slope: {slope:.6f})")
                    else:
                        print(f"🤖 LLM triggered by adaptive advisor (reason: {reason})")
                else:
                    print(f"🤖 LLM decision (alpha={self.alpha})")
                
                return llm_hyperparams, decision_source
                
            except Exception as e:
                print(f"🚨 LLM generation failed: {e}")
                self.rl_decision_count += 1
                return rl_hyperparams, "RL (LLM fallback)"
        else:
            self.rl_decision_count += 1
            decision_source = "RL"
            
            if self.use_adaptive_trigger:
                reason = decision_info.get('trigger_reason', 'unknown')
                slope = decision_info.get('learning_slope', 0)
                if slope is not None:
                    print(f"🎯 RL continuing (reason: {reason}, slope: {slope:.6f})")
                else:
                    print(f"🎯 RL continuing (reason: {reason})")
            else:
                print(f"🎯 RL decision (alpha={self.alpha})")
            
            return rl_hyperparams, decision_source
    
    def update_history(self, metrics: Dict[str, float], hyperparams: Dict[str, Any], step: int = 0):
        """Update training history"""
        self.training_history.append({
            'metrics': metrics.copy(),
            'hyperparams': hyperparams.copy(),
            'step': step
        })
        
        # Keep only recent history
        if len(self.training_history) > 10:
            self.training_history = self.training_history[-10:]
    
    def get_usage_stats(self) -> Dict:
        """Get decision usage statistics"""
        total = self.rl_decision_count + self.llm_decision_count
        if total == 0:
            stats = {'rl_pct': 0, 'llm_pct': 0, 'total': 0}
        else:
            stats = {
                'rl_count': self.rl_decision_count,
                'llm_count': self.llm_decision_count,
                'rl_pct': (self.rl_decision_count / total) * 100,
                'llm_pct': (self.llm_decision_count / total) * 100,
                'total': total
            }
        
        # Add adaptive advisor statistics if available
        if self.use_adaptive_trigger and self.adaptive_advisor is not None:
            advisor_stats = self.adaptive_advisor.get_statistics()
            stats.update({
                'adaptive_mode': True,
                'slope_threshold': advisor_stats['slope_threshold'],
                'trigger_rate': advisor_stats['trigger_rate'],
                'avg_slope': advisor_stats.get('avg_slope', 0),
                'recent_slope': advisor_stats.get('recent_slope', 0)
            })
        else:
            stats.update({
                'adaptive_mode': False,
                'fixed_alpha': self.alpha
            })
        
        return stats