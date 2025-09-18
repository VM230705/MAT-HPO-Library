"""
Ollama LLM Client implementation for MAT_HPO_LIB

Provides integration with Ollama models for hyperparameter optimization.
This is a refactored version of the original OllamaLLMClient for better modularity.
"""

import requests
import json
from typing import Dict, List, Any
from .base_llm_client import BaseLLMClient, DefaultJSONParser
from .dataset_info_reader import get_enhanced_dataset_info, get_dataset_recommendations


class OllamaLLMClient(BaseLLMClient):
    """
    Ollama LLM Client for MAT_HPO_LIB

    Provides unified hyperparameter generation for all 3 agents using
    Large Language Models through the Ollama API.

    Example usage:
        client = OllamaLLMClient(
            model_name="llama3.2:3b",
            base_url="http://localhost:11434"
        )
    """

    def __init__(self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama LLM Client

        Args:
            model_name: Ollama model name (default: llama3.2:3b)
            base_url: Ollama service URL (default: http://localhost:11434)
        """
        super().__init__()

        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

        # Test connection
        self._test_connection()

    def _test_connection(self) -> None:
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
            print("Make sure Ollama is running: 'ollama serve' and model is pulled: f'ollama pull {self.model_name}'")

    def _call_llm_api(self, prompt: str, **kwargs) -> str:
        """
        Call Ollama API

        Args:
            prompt: The prompt to send
            **kwargs: Additional parameters (temperature, top_p, etc.)

        Returns:
            Response text from the model
        """
        # Default parameters
        request_params = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.9),
                "num_predict": kwargs.get('max_tokens', 500)
            }
        }

        response = requests.post(self.api_url, json=request_params, timeout=30)

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        result = response.json()
        if "response" not in result:
            raise Exception("Empty response from Ollama API")

        return result["response"]

    def _parse_hyperparameters(self, llm_response: str, hyperparameter_space) -> Dict[str, Any]:
        """Parse Ollama response into hyperparameter dictionary"""
        try:
            # Use default JSON parser
            parsed_json = DefaultJSONParser.extract_json_from_response(llm_response)

            # Validate against hyperparameter space
            return self._validate_parameters(parsed_json, hyperparameter_space)

        except Exception as e:
            print(f"Error parsing Ollama response: {e}")
            print(f"Response: {llm_response}")
            raise

    def _create_prompt(self, hyperparameter_space, dataset_info: Dict,
                      training_history: List[Dict], previous_error: str = None) -> str:
        """
        Create Ollama-optimized prompt for hyperparameter generation

        Enhanced with dataset-specific information and smart guidance.
        """
        # Get dataset name for enhanced info
        dataset_name = dataset_info.get('name', 'Unknown')
        enhanced_info = get_enhanced_dataset_info(dataset_name)
        recommendations = get_dataset_recommendations(dataset_name)

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

        # Smart guidance
        if enhanced_info and recommendations:
            guidance = self._get_smart_guidance(enhanced_info, recommendations)
        else:
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