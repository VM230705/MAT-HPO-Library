"""
Base LLM Client abstraction for MAT_HPO_LIB - User-configurable LLM integration

This module provides a base abstract class that users can inherit to integrate
their own LLM models (OpenAI, Claude, local models, etc.) into the MAT_HPO_LIB
hyperparameter optimization framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import json
import numpy as np


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients in MAT_HPO_LIB

    Users should inherit from this class and implement the abstract methods
    to integrate their preferred LLM models into the hyperparameter optimization.

    Example:
        class CustomLLMClient(BaseLLMClient):
            def __init__(self, api_key="your_api_key"):
                super().__init__()
                self.api_key = api_key
                self._test_connection()

            def _test_connection(self):
                # Test your model connection
                pass

            def _call_llm_api(self, prompt: str) -> str:
                # Call your LLM API
                return your_llm_response

            def _parse_hyperparameters(self, response: str, hyperparameter_space) -> Dict[str, Any]:
                # Parse your LLM's response format
                return parsed_params
    """

    def __init__(self):
        """Initialize base LLM client"""
        self.conversation_logger = None
        self.current_step = 0
        self.generation_count = 0
        self.success_count = 0
        self.failure_count = 0

    @abstractmethod
    def _test_connection(self) -> None:
        """
        Test connection to the LLM service

        Should print connection status and raise exception if connection fails
        """
        pass

    @abstractmethod
    def _call_llm_api(self, prompt: str, **kwargs) -> str:
        """
        Make API call to LLM service

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional model-specific parameters

        Returns:
            Raw response string from the LLM

        Raises:
            Exception: If the API call fails
        """
        pass

    @abstractmethod
    def _parse_hyperparameters(self, llm_response: str, hyperparameter_space) -> Dict[str, Any]:
        """
        Parse LLM response into hyperparameter dictionary

        Args:
            llm_response: Raw response from the LLM
            hyperparameter_space: HyperparameterSpace object defining valid parameters

        Returns:
            Dictionary mapping hyperparameter names to values

        Raises:
            Exception: If parsing fails
        """
        pass

    def generate_hyperparameters(self,
                               hyperparameter_space,
                               dataset_info: Dict,
                               training_history: List[Dict],
                               max_retries: int = 3,
                               **llm_kwargs) -> Dict[str, Any]:
        """
        Generate hyperparameters using LLM (with retry logic)

        This is the main method called by the optimizer. It handles retry logic,
        logging, and fallback to safe defaults if all attempts fail.

        Args:
            hyperparameter_space: HyperparameterSpace object defining the search space
            dataset_info: Dataset characteristics
            training_history: Previous training results
            max_retries: Maximum number of retry attempts
            **llm_kwargs: Additional arguments passed to _call_llm_api

        Returns:
            Dictionary of hyperparameter names to values
        """
        self.generation_count += 1
        last_error = None

        for attempt in range(max_retries):
            try:
                # Create prompt
                prompt = self._create_prompt(hyperparameter_space, dataset_info, training_history,
                                           previous_error=last_error if attempt > 0 else None)

                # Call LLM API
                response = self._call_llm_api(prompt, **llm_kwargs)

                # Parse response
                parsed_params = self._parse_hyperparameters(response, hyperparameter_space)

                # Validate parsed parameters
                validated_params = self._validate_parameters(parsed_params, hyperparameter_space)

                # Log successful generation
                self._log_success(attempt + 1, prompt, response, validated_params, dataset_info, training_history)

                self.success_count += 1
                return validated_params

            except Exception as e:
                last_error = str(e)
                self._log_attempt_failure(attempt + 1, last_error)
                continue

        # All attempts failed - use fallback
        self.failure_count += 1
        print("❌ ALL LLM ATTEMPTS FAILED - Using fallback hyperparameters")
        fallback_params = self._fallback_hyperparameters(hyperparameter_space)

        # Log complete failure
        self._log_complete_failure(last_error, fallback_params)

        return fallback_params

    def _create_prompt(self, hyperparameter_space, dataset_info: Dict,
                      training_history: List[Dict], previous_error: str = None) -> str:
        """
        Create prompt for hyperparameter generation

        This method can be overridden by subclasses to customize the prompt format
        for specific LLM models.
        """
        # Dataset context
        dataset_context = self._format_dataset_context(dataset_info)

        # Training history context
        history_context = self._format_training_history(training_history)

        # Hyperparameter space description
        param_descriptions = self._format_parameter_space(hyperparameter_space)

        # Error feedback if this is a retry
        error_feedback = ""
        if previous_error:
            error_feedback = f"""
❌ PREVIOUS ATTEMPT FAILED: {previous_error}

CRITICAL: You must provide hyperparameters in JSON format as specified below.
"""

        # Smart guidance based on dataset
        guidance = self._get_dataset_guidance(dataset_info)

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
  "parameter_name_1": value1,
  "parameter_name_2": value2,
  ...
}}

Provide ONLY the JSON object with hyperparameter names and their values. No explanations.
"""

        return prompt

    def _format_dataset_context(self, dataset_info: Dict) -> str:
        """Format dataset information for prompt"""
        return f"""
Dataset: {dataset_info.get('name', 'Unknown')}
Training Samples: {dataset_info.get('total_samples', 'Unknown')}
Features: {dataset_info.get('num_features', 'Unknown')}
Classes: {dataset_info.get('num_classes', 'Unknown')}
"""

    def _format_training_history(self, training_history: List[Dict]) -> str:
        """Format training history for prompt"""
        if not training_history:
            return ""

        history_context = "Recent Training Results:\n"
        recent = training_history[-3:]  # Last 3 results
        for i, result in enumerate(recent):
            metrics = result.get('metrics', {})
            f1 = metrics.get('f1', result.get('f1', 0))
            auc = metrics.get('auc', result.get('auc', 0))
            gmean = metrics.get('gmean', result.get('gmean', 0))
            history_context += f"Step {len(training_history)-len(recent)+i+1}: F1={f1:.3f}, AUC={auc:.3f}, GMean={gmean:.3f}\n"

        return history_context

    def _format_parameter_space(self, hyperparameter_space) -> List[str]:
        """Format hyperparameter space for prompt"""
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

        return param_descriptions

    def _get_dataset_guidance(self, dataset_info: Dict) -> str:
        """Generate guidance based on dataset characteristics (can be overridden)"""
        total_samples = dataset_info.get('total_samples', 1000)
        num_classes = dataset_info.get('num_classes', 2)

        guidance_parts = []

        # Dataset size guidance
        if total_samples < 1000:
            guidance_parts.append("SMALL DATASET: Use conservative parameters to prevent overfitting.")
        elif total_samples < 10000:
            guidance_parts.append("MEDIUM DATASET: Balance model capacity with generalization.")
        else:
            guidance_parts.append("LARGE DATASET: Can use larger models and complex architectures.")

        # Multi-class guidance
        if num_classes > 2:
            guidance_parts.append(f"MULTI-CLASS PROBLEM ({num_classes} classes): Consider class imbalance.")

        return " ".join(guidance_parts) if guidance_parts else "Use balanced hyperparameters appropriate for the dataset."

    def _validate_parameters(self, params: Dict[str, Any], hyperparameter_space) -> Dict[str, Any]:
        """Validate and bound parameters to search space"""
        validated_params = {}

        if hasattr(hyperparameter_space, 'parameters'):
            for param_name, param_info in hyperparameter_space.parameters.items():
                if param_name in params:
                    value = params[param_name]
                    param_type = param_info.get('type', 'continuous')

                    if param_type == 'continuous':
                        min_val, max_val = param_info['bounds']
                        validated_params[param_name] = max(min_val, min(max_val, float(value)))
                    elif param_type == 'discrete':
                        choices = param_info['choices']
                        if value in choices:
                            validated_params[param_name] = value
                        else:
                            # Choose closest or default
                            validated_params[param_name] = choices[0]
                    else:
                        validated_params[param_name] = value

        return validated_params

    def _fallback_hyperparameters(self, hyperparameter_space) -> Dict[str, Any]:
        """Generate safe fallback hyperparameters when LLM fails"""
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

    def _log_success(self, attempt: int, prompt: str, response: str, parsed_params: Dict,
                    dataset_info: Dict, training_history: List[Dict]) -> None:
        """Log successful LLM generation"""
        if self.conversation_logger:
            self.conversation_logger.log_llm_conversation(
                step=self.current_step,
                attempt=attempt,
                prompt=prompt,
                response=response,
                parse_success=True,
                parsed_params=parsed_params,
                dataset_info=dataset_info,
                training_history=training_history
            )

    def _log_attempt_failure(self, attempt: int, error: str) -> None:
        """Log individual attempt failure"""
        print(f"🔄 Attempt {attempt}/3 failed: {error}")

    def _log_complete_failure(self, last_error: str, fallback_params: Dict) -> None:
        """Log complete failure and fallback usage"""
        if self.conversation_logger:
            self.conversation_logger.log_failure_summary(
                step=self.current_step,
                all_attempts=[{"error_message": last_error}],
                fallback_params=fallback_params
            )

    def get_statistics(self) -> Dict:
        """Get LLM client statistics"""
        total_attempts = self.generation_count
        success_rate = self.success_count / max(total_attempts, 1)

        return {
            'total_generations': self.generation_count,
            'successful_generations': self.success_count,
            'failed_generations': self.failure_count,
            'success_rate': success_rate,
            'model_name': getattr(self, 'model_name', 'Unknown')
        }


class DefaultJSONParser:
    """Default JSON parser for common LLM response formats"""

    @staticmethod
    def extract_json_from_response(response: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response using multiple strategies

        Tries different approaches to find JSON in various response formats
        """
        import re

        # Strategy 1: Look for complete JSON object
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Strategy 2: Look for JSON between code blocks
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Strategy 3: Extract key-value pairs manually
        manual_json = {}
        kv_pattern = r'"([^"]+)":\s*([^,}\n]+)'
        matches = re.findall(kv_pattern, response)

        for key, value in matches:
            try:
                # Try to parse as number
                if '.' in value:
                    manual_json[key] = float(value.strip())
                else:
                    manual_json[key] = int(value.strip())
            except ValueError:
                # Keep as string, remove quotes
                manual_json[key] = value.strip().strip('"')

        if manual_json:
            return manual_json

        raise ValueError("No valid JSON found in LLM response")