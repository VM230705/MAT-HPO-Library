"""
Enhanced Base Environment for Time Series with LLM Integration
Extends BaseEnvironment with automatic dataset analysis and LLM context generation
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List
from .base_environment import BaseEnvironment
from ..llm import get_universal_dataset_info, TimeSeriesAnalyzer


class TimeSeriesEnvironment(BaseEnvironment):
    """
    Enhanced environment for time series datasets with automatic analysis
    and LLM context generation
    """

    def __init__(self,
                 name: str = "TimeSeriesTask",
                 validation_split: float = 0.1,
                 dataset_name: str = "Unknown",
                 enable_auto_analysis: bool = True,
                 csv_info_path: Optional[str] = None):
        """
        Initialize enhanced time series environment

        Args:
            name: Environment name
            validation_split: Validation split ratio
            dataset_name: Name of the dataset for LLM context
            enable_auto_analysis: Enable automatic dataset analysis
            csv_info_path: Path to dataset info CSV (optional)
        """
        super().__init__(name=name, validation_split=validation_split)

        self.dataset_name = dataset_name
        self.enable_auto_analysis = enable_auto_analysis
        self.csv_info_path = csv_info_path

        # Data structure attributes
        self.is_variable_length = False
        self.variable_dim = None
        self.min_sequences = None
        self.max_sequences = None
        self.avg_sequences = None

        # Dataset analysis results
        self.dataset_info = None
        self.llm_context = None

        # Time series specific data
        self.X_raw = None
        self.y_raw = None
        self.is_multivariate = None
        self.sequence_length = None
        self.n_features = None

    def set_dataset(self,
                   X: Union[np.ndarray, Any],
                   y: Optional[Union[np.ndarray, Any]] = None,
                   dataset_name: Optional[str] = None):
        """
        Set dataset and perform automatic analysis

        Args:
            X: Time series data
            y: Labels (optional)
            dataset_name: Dataset name (overrides constructor)
        """
        self.X_raw = X
        self.y_raw = y

        if dataset_name:
            self.dataset_name = dataset_name

        # Analyze data structure
        if isinstance(X, np.ndarray):
            if X.ndim == 1 and X.dtype == object:
                # Variable-length sequence data (e.g., SPNV2 snippets)
                self._handle_variable_length_data(X)
            elif X.ndim == 2:
                self.is_multivariate = False
                self.n_features = 1
                self.sequence_length = X.shape[1]
            elif X.ndim == 3:
                self.is_multivariate = True
                self.n_features = X.shape[1]
                self.sequence_length = X.shape[2]
            else:
                raise ValueError(f"Unsupported data shape: {X.shape}")
        else:
            raise ValueError(f"Unsupported data type: {type(X)}")

        # Perform automatic dataset analysis for LLM context
        if self.enable_auto_analysis:
            self._analyze_dataset()

    def _handle_variable_length_data(self, X):
        """Handle variable-length sequence data (object arrays)"""
        print(f"🔍 Detecting variable-length sequence data: {X.shape}")

        # Analyze the structure of variable-length data
        if len(X) == 0:
            raise ValueError("Empty dataset")

        # Sample a few examples to understand the structure
        sample_sizes = []
        sample_shapes = []

        for i in range(min(10, len(X))):
            if hasattr(X[i], 'shape'):
                sample_shapes.append(X[i].shape)
                if X[i].ndim >= 1:
                    sample_sizes.append(X[i].shape[0])  # First dimension (number of sequences/snippets)

        print(f"   Sample shapes: {sample_shapes[:3]}...")
        print(f"   Sample sizes (first dim): {sample_sizes[:3]}...")

        if sample_shapes:
            # Determine data characteristics from samples
            typical_shape = sample_shapes[0]

            if len(typical_shape) == 3:
                # Format: (num_snippets, sequence_length, features)
                self.is_variable_length = True
                self.is_multivariate = True
                self.n_features = typical_shape[2]  # Last dimension
                self.sequence_length = typical_shape[1]  # Middle dimension
                self.variable_dim = 0  # First dimension varies

                # Statistics
                self.min_sequences = min(sample_sizes) if sample_sizes else 1
                self.max_sequences = max(sample_sizes) if sample_sizes else 1
                self.avg_sequences = sum(sample_sizes) / len(sample_sizes) if sample_sizes else 1

                print(f"   ✅ Variable-length multivariate sequences detected:")
                print(f"      Sequences per sample: {self.min_sequences}-{self.max_sequences} (avg: {self.avg_sequences:.1f})")
                print(f"      Sequence length: {self.sequence_length}")
                print(f"      Features: {self.n_features}")

            elif len(typical_shape) == 2:
                # Format: (num_sequences, features)
                self.is_variable_length = True
                self.is_multivariate = typical_shape[1] > 1
                self.n_features = typical_shape[1]
                self.sequence_length = None  # Variable
                self.variable_dim = 0

                print(f"   ✅ Variable-length feature sequences detected:")
                print(f"      Sequences per sample: {min(sample_sizes)}-{max(sample_sizes)}")
                print(f"      Features: {self.n_features}")

            else:
                # Fallback: treat as generic variable-length
                self.is_variable_length = True
                self.is_multivariate = False
                self.n_features = 1
                self.sequence_length = None
                self.variable_dim = 0

                print(f"   ⚠️  Generic variable-length data detected")
        else:
            raise ValueError("Could not analyze variable-length data structure")

    def _analyze_dataset(self):
        """Perform automatic dataset analysis"""
        try:
            # Get universal dataset info (CSV first, then auto-analysis)
            info = get_universal_dataset_info(
                dataset_name=self.dataset_name,
                X=self.X_raw,
                y=self.y_raw
            )

            self.dataset_info = info["dataset_info"]
            self.llm_context = info["llm_context"]

            print(f"📊 Dataset Analysis: {info['source']}")
            print(f"🤖 LLM Context: {self.llm_context[:100]}...")

        except Exception as e:
            print(f"⚠️ Dataset analysis failed: {e}")
            self.llm_context = f"Dataset: {self.dataset_name} (analysis failed)"

    def get_llm_context(self) -> str:
        """Get LLM context for prompts"""
        if self.llm_context is None:
            return f"Dataset: {self.dataset_name} (no analysis available)"
        return self.llm_context

    def get_dataset_recommendations(self) -> Dict[str, Any]:
        """Get hyperparameter recommendations based on dataset analysis"""
        if not self.dataset_info:
            return {}

        # Extract recommendations from analysis
        if "recommendations" in self.dataset_info:
            return self.dataset_info["recommendations"]

        # Fallback basic recommendations
        return {
            "model_architecture": {"suggested_model_types": ["LSTM", "CNN-LSTM"]},
            "training_parameters": {
                "batch_size_range": [32, 128],
                "learning_rate_range": [0.0001, 0.01]
            },
            "optimization_focus": ["performance_tuning"]
        }

    def get_complexity_metrics(self) -> Dict[str, Any]:
        """Get dataset complexity metrics"""
        if not self.dataset_info:
            return {}

        # Return complexity information for HPO guidance
        if "complexity_metrics" in self.dataset_info:
            return self.dataset_info["complexity_metrics"]

        # Basic complexity from data shape
        if self.X_raw is not None:
            n_samples = len(self.X_raw)
            return {
                "sample_size_category": "small" if n_samples < 1000 else "medium" if n_samples < 10000 else "large",
                "sequence_length_category": "short" if self.sequence_length < 100 else "medium" if self.sequence_length < 500 else "long",
                "feature_complexity_category": "simple" if self.n_features == 1 else "moderate" if self.n_features < 10 else "complex"
            }

        return {}

    def should_use_class_weights(self) -> bool:
        """Determine if class weights should be used based on dataset analysis"""
        if not self.dataset_info or "class_distribution" not in self.dataset_info:
            return False

        class_dist = self.dataset_info["class_distribution"]
        if class_dist and "max_imbalance_ratio" in class_dist:
            return class_dist["max_imbalance_ratio"] > 2.0

        return False

    def get_suggested_class_weights(self) -> Optional[Dict[int, float]]:
        """Get suggested class weights based on dataset analysis"""
        if not self.should_use_class_weights():
            return None

        if self.dataset_info and "class_distribution" in self.dataset_info:
            class_dist = self.dataset_info["class_distribution"]
            if "class_counts" in class_dist:
                # Calculate inverse frequency weights
                class_counts = class_dist["class_counts"]
                total_samples = sum(class_counts.values())
                n_classes = len(class_counts)

                weights = {}
                for class_id, count in class_counts.items():
                    weights[class_id] = total_samples / (n_classes * count)

                return weights

        return None

    # Abstract methods that subclasses must implement
    def load_data(self):
        """Load and prepare data - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement load_data()")

    def create_model(self, hyperparams: Dict[str, Any]):
        """Create model with given hyperparameters - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement create_model()")

    def train_evaluate(self, model, hyperparams: Dict[str, Any]) -> Dict[str, float]:
        """Train and evaluate model - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement train_evaluate()")

    def compute_reward(self, metrics: Dict[str, float]) -> float:
        """Compute reward from metrics - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement compute_reward()")