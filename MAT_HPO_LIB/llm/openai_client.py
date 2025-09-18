"""
OpenAI LLM Client implementation for MAT_HPO_LIB

Provides integration with OpenAI's GPT models for hyperparameter optimization.
Requires 'openai' package: pip install openai
"""

from typing import Dict, List, Any
import json
from .base_llm_client import BaseLLMClient, DefaultJSONParser

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI package not available. Install with: pip install openai")


class OpenAILLMClient(BaseLLMClient):
    """
    OpenAI LLM Client for MAT_HPO_LIB

    Supports GPT-3.5-turbo, GPT-4, and other OpenAI models for hyperparameter
    generation with configurable parameters.

    Example usage:
        client = OpenAILLMClient(
            api_key="your_api_key",
            model="gpt-4",
            temperature=0.7
        )
    """

    def __init__(self,
                 api_key: str = None,
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0.7,
                 max_tokens: int = 500,
                 top_p: float = 0.9,
                 base_url: str = None):
        """
        Initialize OpenAI LLM Client

        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: Model name (gpt-3.5-turbo, gpt-4, etc.)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            base_url: Custom API base URL (for compatible services)
        """
        super().__init__()

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is required. Install with: pip install openai")

        self.api_key = api_key
        self.model = model
        self.model_name = model  # For statistics
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        # Initialize OpenAI client
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if base_url:
            client_kwargs['base_url'] = base_url

        self.client = openai.OpenAI(**client_kwargs)

        # Test connection
        self._test_connection()

    def _test_connection(self) -> None:
        """Test connection to OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                timeout=10
            )
            if response.choices:
                print(f"✅ Connected to OpenAI {self.model}")
            else:
                print(f"⚠️ OpenAI connection issue: Empty response")
        except Exception as e:
            print(f"❌ Failed to connect to OpenAI: {e}")
            print("Make sure your API key is set and you have sufficient credits")
            raise

    def _call_llm_api(self, prompt: str, **kwargs) -> str:
        """
        Call OpenAI API

        Args:
            prompt: The prompt to send
            **kwargs: Additional parameters (override instance defaults)

        Returns:
            Response text from the model
        """
        # Merge kwargs with instance defaults
        call_params = {
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'top_p': kwargs.get('top_p', self.top_p)
        }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **call_params
        )

        if not response.choices:
            raise ValueError("Empty response from OpenAI API")

        return response.choices[0].message.content

    def _parse_hyperparameters(self, llm_response: str, hyperparameter_space) -> Dict[str, Any]:
        """Parse OpenAI response into hyperparameter dictionary"""
        try:
            # Use default JSON parser
            parsed_json = DefaultJSONParser.extract_json_from_response(llm_response)

            # Validate against hyperparameter space
            return self._validate_parameters(parsed_json, hyperparameter_space)

        except Exception as e:
            print(f"Error parsing OpenAI response: {e}")
            print(f"Response: {llm_response}")
            raise

    def _get_dataset_guidance(self, dataset_info: Dict) -> str:
        """Enhanced guidance for OpenAI models"""
        base_guidance = super()._get_dataset_guidance(dataset_info)

        # Add OpenAI-specific guidance
        openai_guidance = []

        total_samples = dataset_info.get('total_samples', 1000)
        num_features = dataset_info.get('num_features', 1)

        if total_samples < 5000 and num_features > 100:
            openai_guidance.append("HIGH DIMENSIONALITY + SMALL DATA: Use strong regularization and feature selection.")

        if dataset_info.get('num_classes', 2) > 10:
            openai_guidance.append("MANY CLASSES: Consider hierarchical classification strategies.")

        combined_guidance = base_guidance
        if openai_guidance:
            combined_guidance += " " + " ".join(openai_guidance)

        return combined_guidance


class AzureOpenAILLMClient(OpenAILLMClient):
    """
    Azure OpenAI LLM Client for MAT_HPO_LIB

    Specialized client for Azure OpenAI services with proper endpoint handling.

    Example usage:
        client = AzureOpenAILLMClient(
            api_key="your_api_key",
            endpoint="https://your-resource.openai.azure.com/",
            deployment_name="gpt-4",
            api_version="2024-02-15-preview"
        )
    """

    def __init__(self,
                 api_key: str,
                 endpoint: str,
                 deployment_name: str,
                 api_version: str = "2024-02-15-preview",
                 temperature: float = 0.7,
                 max_tokens: int = 500,
                 top_p: float = 0.9):
        """
        Initialize Azure OpenAI LLM Client

        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL
            deployment_name: Deployment name in Azure
            api_version: API version
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is required. Install with: pip install openai")

        # Initialize base class attributes before calling super()
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.endpoint = endpoint
        self.api_version = api_version
        self.model = deployment_name  # For compatibility
        self.model_name = deployment_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        # Initialize base class
        BaseLLMClient.__init__(self)

        # Initialize Azure OpenAI client
        self.client = openai.AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )

        # Test connection
        self._test_connection()

    def _call_llm_api(self, prompt: str, **kwargs) -> str:
        """Call Azure OpenAI API"""
        call_params = {
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'top_p': kwargs.get('top_p', self.top_p)
        }

        response = self.client.chat.completions.create(
            model=self.deployment_name,  # Use deployment name for Azure
            messages=[{"role": "user", "content": prompt}],
            **call_params
        )

        if not response.choices:
            raise ValueError("Empty response from Azure OpenAI API")

        return response.choices[0].message.content