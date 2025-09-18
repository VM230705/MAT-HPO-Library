"""
Anthropic Claude LLM Client implementation for MAT_HPO_LIB

Provides integration with Anthropic's Claude models for hyperparameter optimization.
Requires 'anthropic' package: pip install anthropic
"""

from typing import Dict, List, Any
import json
from .base_llm_client import BaseLLMClient, DefaultJSONParser

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️ Anthropic package not available. Install with: pip install anthropic")


class AnthropicLLMClient(BaseLLMClient):
    """
    Anthropic Claude LLM Client for MAT_HPO_LIB

    Supports Claude-3 (Haiku, Sonnet, Opus) and Claude-3.5 models for hyperparameter
    generation with configurable parameters.

    Example usage:
        client = AnthropicLLMClient(
            api_key="your_api_key",
            model="claude-3-5-sonnet-20241022",
            temperature=0.7
        )
    """

    def __init__(self,
                 api_key: str = None,
                 model: str = "claude-3-5-sonnet-20241022",
                 temperature: float = 0.7,
                 max_tokens: int = 500,
                 top_p: float = 0.9,
                 base_url: str = None):
        """
        Initialize Anthropic LLM Client

        Args:
            api_key: Anthropic API key (if None, will use ANTHROPIC_API_KEY env var)
            model: Model name (claude-3-5-sonnet-20241022, claude-3-opus-20240229, etc.)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            base_url: Custom API base URL (for compatible services)
        """
        super().__init__()

        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package is required. Install with: pip install anthropic")

        self.api_key = api_key
        self.model = model
        self.model_name = model  # For statistics
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        # Initialize Anthropic client
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if base_url:
            client_kwargs['base_url'] = base_url

        self.client = anthropic.Anthropic(**client_kwargs)

        # Test connection
        self._test_connection()

    def _test_connection(self) -> None:
        """Test connection to Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=5,
                messages=[{"role": "user", "content": "Hello"}],
                timeout=10
            )
            if response.content:
                print(f"✅ Connected to Anthropic {self.model}")
            else:
                print(f"⚠️ Anthropic connection issue: Empty response")
        except Exception as e:
            print(f"❌ Failed to connect to Anthropic: {e}")
            print("Make sure your API key is set and you have sufficient credits")
            raise

    def _call_llm_api(self, prompt: str, **kwargs) -> str:
        """
        Call Anthropic API

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

        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **call_params
        )

        if not response.content:
            raise ValueError("Empty response from Anthropic API")

        # Extract text from response
        text_content = ""
        for content_block in response.content:
            if hasattr(content_block, 'text'):
                text_content += content_block.text

        return text_content

    def _parse_hyperparameters(self, llm_response: str, hyperparameter_space) -> Dict[str, Any]:
        """Parse Anthropic response into hyperparameter dictionary"""
        try:
            # Use default JSON parser
            parsed_json = DefaultJSONParser.extract_json_from_response(llm_response)

            # Validate against hyperparameter space
            return self._validate_parameters(parsed_json, hyperparameter_space)

        except Exception as e:
            print(f"Error parsing Anthropic response: {e}")
            print(f"Response: {llm_response}")
            raise

    def _create_prompt(self, hyperparameter_space, dataset_info: Dict,
                      training_history: List[Dict], previous_error: str = None) -> str:
        """
        Create Claude-optimized prompt for hyperparameter generation

        Claude responds well to structured, clear instructions with examples.
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
<error>
PREVIOUS ATTEMPT FAILED: {previous_error}

You must provide hyperparameters in the exact JSON format specified below.
</error>
"""

        # Smart guidance based on dataset
        guidance = self._get_dataset_guidance(dataset_info)

        # Create example for Claude
        example_params = self._create_example_parameters(hyperparameter_space)

        prompt = f"""You are an expert in hyperparameter optimization for machine learning models.

<dataset_info>
{dataset_context}
</dataset_info>

<training_history>
{history_context}
</training_history>

<hyperparameter_space>
{chr(10).join(param_descriptions)}
</hyperparameter_space>

<guidance>
{guidance}
</guidance>

{error_feedback}

<task>
Based on the dataset characteristics and training history, provide optimal hyperparameters.

Your response must be a JSON object containing hyperparameter names and their values.
</task>

<example_format>
{example_params}
</example_format>

<instructions>
- Analyze the dataset characteristics carefully
- Consider the training history trends
- Stay within the specified parameter bounds
- Return ONLY the JSON object, no explanations
- Use the exact parameter names from the search space
</instructions>

JSON response:"""

        return prompt

    def _create_example_parameters(self, hyperparameter_space) -> str:
        """Create example JSON format for Claude"""
        example = {}
        if hasattr(hyperparameter_space, 'parameters'):
            for param_name, param_info in list(hyperparameter_space.parameters.items())[:3]:  # Show first 3
                param_type = param_info.get('type', 'continuous')
                if param_type == 'continuous':
                    min_val, max_val = param_info['bounds']
                    example[param_name] = (min_val + max_val) / 2
                elif param_type == 'discrete':
                    choices = param_info['choices']
                    example[param_name] = choices[0]

        return json.dumps(example, indent=2)

    def _get_dataset_guidance(self, dataset_info: Dict) -> str:
        """Enhanced guidance optimized for Claude's reasoning"""
        base_guidance = super()._get_dataset_guidance(dataset_info)

        # Add Claude-specific analytical guidance
        claude_guidance = []

        total_samples = dataset_info.get('total_samples', 1000)
        num_features = dataset_info.get('num_features', 1)
        num_classes = dataset_info.get('num_classes', 2)

        # Analytical reasoning for Claude
        if total_samples < 1000:
            claude_guidance.append("With limited data, prioritize regularization over model complexity.")

        if num_features > total_samples:
            claude_guidance.append("High-dimensional data requires careful feature selection and regularization.")

        if num_classes > 2:
            claude_guidance.append(f"Multi-class problem with {num_classes} classes may benefit from class balancing.")

        # Combine guidances
        combined_guidance = base_guidance
        if claude_guidance:
            combined_guidance += " Additional considerations: " + " ".join(claude_guidance)

        return combined_guidance


class ClaudeLegacyClient(AnthropicLLMClient):
    """
    Legacy Claude client for older API versions

    Maintains compatibility with earlier Claude API formats while using
    the same underlying optimization logic.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with legacy model defaults"""
        if 'model' not in kwargs:
            kwargs['model'] = "claude-3-haiku-20240307"  # Default to Haiku for cost efficiency

        super().__init__(*args, **kwargs)
        print(f"⚠️ Using legacy Claude client with {self.model}")

    def _create_prompt(self, hyperparameter_space, dataset_info: Dict,
                      training_history: List[Dict], previous_error: str = None) -> str:
        """Simplified prompt for legacy Claude models"""
        # Use base class prompt format for simpler models
        return super(AnthropicLLMClient, self)._create_prompt(
            hyperparameter_space, dataset_info, training_history, previous_error
        )