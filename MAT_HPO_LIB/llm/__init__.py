"""
LLM integration components for MAT_HPO_LIB

This module provides Large Language Model integration capabilities for the
MAT_HPO_LIB library, enabling adaptive hyperparameter suggestions through
LLM-assisted optimization.

Key Components:
- BaseLLMClient: Abstract base class for custom LLM integrations
- OllamaLLMClient: LLM client for Ollama integration
- OpenAILLMClient: LLM client for OpenAI models
- AnthropicLLMClient: LLM client for Anthropic Claude models
- LLMHyperparameterMixer: Original mixing RL and LLM suggestions (legacy)
- EnhancedLLMHyperparameterMixer: Advanced mixer with user-configurable clients
- AdaptiveAdvisor: Adaptive triggering mechanism based on LLaPipe paper
- DatasetInfoReader: Dataset-aware prompt generation
- LLMConversationLogger: Comprehensive logging of LLM interactions
"""

# Core interfaces and base classes
from .base_llm_client import BaseLLMClient, DefaultJSONParser

# LLM client implementations
from .ollama_client import OllamaLLMClient
try:
    from .openai_client import OpenAILLMClient, AzureOpenAILLMClient
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAILLMClient = None
    AzureOpenAILLMClient = None

# Only attempt to import Anthropic if explicitly requested (to avoid warnings)
ANTHROPIC_AVAILABLE = False
AnthropicLLMClient = None
ClaudeLegacyClient = None

# Legacy components (maintained for backward compatibility)
from .llm_client import LLMHyperparameterMixer

# Enhanced components
from .enhanced_mixer import EnhancedLLMHyperparameterMixer, AdaptiveAlphaController

# Supporting components
from .adaptive_advisor import LLaPipeAdaptiveAdvisor, PerformanceMetricCalculator
from .dataset_info_reader import (
    DatasetInfoReader, get_enhanced_dataset_info, get_dataset_recommendations,
    analyze_time_series_dataset, get_time_series_llm_context, get_universal_dataset_info
)
from .time_series_analyzer import TimeSeriesAnalyzer
from .conversation_logger import LLMConversationLogger

# Build __all__ dynamically based on available packages
__all__ = [
    # Core interfaces
    "BaseLLMClient",
    "DefaultJSONParser",

    # Always available clients
    "OllamaLLMClient",

    # Enhanced mixing components
    "EnhancedLLMHyperparameterMixer",
    "AdaptiveAlphaController",

    # Legacy components
    "LLMHyperparameterMixer",

    # Supporting components
    "LLaPipeAdaptiveAdvisor",
    "PerformanceMetricCalculator",
    "DatasetInfoReader",
    "get_enhanced_dataset_info",
    "get_dataset_recommendations",
    "analyze_time_series_dataset",
    "get_time_series_llm_context",
    "get_universal_dataset_info",
    "TimeSeriesAnalyzer",
    "LLMConversationLogger"
]

# Add optional clients if available
if OPENAI_AVAILABLE:
    __all__.extend(["OpenAILLMClient", "AzureOpenAILLMClient"])

if ANTHROPIC_AVAILABLE:
    __all__.extend(["AnthropicLLMClient", "ClaudeLegacyClient"])