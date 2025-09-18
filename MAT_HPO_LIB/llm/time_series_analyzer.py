"""
Time Series Dataset Analyzer for MAT_HPO_LIB
Automatically extracts time series characteristics for LLM prompts
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    """Automatically analyze time series datasets to extract characteristics for LLM prompts"""

    def __init__(self):
        self.analysis_results = {}

    def analyze_dataset(self,
                       X: Union[np.ndarray, pd.DataFrame],
                       y: Optional[Union[np.ndarray, pd.Series, List]] = None,
                       dataset_name: str = "Unknown",
                       sample_size: int = 1000) -> Dict[str, Any]:
        """
        Comprehensive analysis of time series dataset

        Args:
            X: Time series data (n_samples, n_features, seq_length) or (n_samples, seq_length)
            y: Labels (optional)
            dataset_name: Name of the dataset
            sample_size: Maximum samples to analyze for performance

        Returns:
            Dictionary with comprehensive dataset characteristics
        """
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, list)):
            y = np.array(y)

        # Limit sample size for performance
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
            y_sample = y[indices] if y is not None else None
        else:
            X_sample = X
            y_sample = y

        analysis = {
            "dataset_name": dataset_name,
            "basic_info": self._analyze_basic_info(X_sample, y_sample),
            "temporal_characteristics": self._analyze_temporal_characteristics(X_sample),
            "statistical_properties": self._analyze_statistical_properties(X_sample),
            "class_distribution": self._analyze_class_distribution(y_sample) if y_sample is not None else None,
            "complexity_metrics": self._analyze_complexity(X_sample),
            "recommendations": {}
        }

        # Generate recommendations based on analysis
        analysis["recommendations"] = self._generate_recommendations(analysis)

        self.analysis_results[dataset_name] = analysis
        return analysis

    def _analyze_basic_info(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze basic dataset information"""
        n_samples, *dims = X.shape

        # Handle variable-length sequence data (object arrays)
        if X.ndim == 1 and X.dtype == object:
            # Variable-length sequence data (e.g., SPNV2 snippets)
            return self._analyze_variable_length_basic_info(X, y)

        if len(dims) == 1:
            # Univariate: (n_samples, seq_length)
            seq_length = dims[0]
            n_features = 1
            is_multivariate = False
        elif len(dims) == 2:
            # Multivariate: (n_samples, n_features, seq_length)
            n_features, seq_length = dims
            is_multivariate = True
        else:
            raise ValueError(f"Unsupported data shape: {X.shape}")

        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "sequence_length": seq_length,
            "is_multivariate": is_multivariate,
            "data_shape": X.shape,
            "n_classes": len(np.unique(y)) if y is not None else None,
            "has_missing_values": np.isnan(X).any(),
            "missing_ratio": np.isnan(X).mean() if np.isnan(X).any() else 0.0,
            "is_variable_length": False
        }

    def _analyze_variable_length_basic_info(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze basic info for variable-length sequence data"""
        print(f"🔍 Analyzing variable-length sequence data: {X.shape}")

        n_samples = len(X)

        # Sample a few examples to understand the structure
        sample_shapes = []
        sample_sizes = []

        for i in range(min(10, n_samples)):
            if hasattr(X[i], 'shape'):
                sample_shapes.append(X[i].shape)
                if X[i].ndim >= 1:
                    sample_sizes.append(X[i].shape[0])  # First dimension (number of sequences/snippets)

        if not sample_shapes:
            raise ValueError("Could not analyze variable-length data structure")

        # Determine characteristics from samples
        typical_shape = sample_shapes[0]

        if len(typical_shape) == 3:
            # Format: (num_snippets, sequence_length, features)
            n_features = typical_shape[2]
            sequence_length = typical_shape[1]
            is_multivariate = True
            variable_dim = 0  # First dimension varies

            min_sequences = min(sample_sizes) if sample_sizes else 1
            max_sequences = max(sample_sizes) if sample_sizes else 1
            avg_sequences = sum(sample_sizes) / len(sample_sizes) if sample_sizes else 1

            print(f"   ✅ Variable-length multivariate sequences detected:")
            print(f"      Sequences per sample: {min_sequences}-{max_sequences} (avg: {avg_sequences:.1f})")
            print(f"      Sequence length: {sequence_length}")
            print(f"      Features: {n_features}")

        elif len(typical_shape) == 2:
            # Format: (num_sequences, features)
            n_features = typical_shape[1]
            sequence_length = None  # Variable
            is_multivariate = n_features > 1
            variable_dim = 0

            min_sequences = min(sample_sizes) if sample_sizes else 1
            max_sequences = max(sample_sizes) if sample_sizes else 1
            avg_sequences = sum(sample_sizes) / len(sample_sizes) if sample_sizes else 1

            print(f"   ✅ Variable-length feature sequences detected:")
            print(f"      Sequences per sample: {min_sequences}-{max_sequences}")
            print(f"      Features: {n_features}")

        else:
            # Fallback: treat as generic variable-length
            n_features = 1
            sequence_length = None
            is_multivariate = False
            variable_dim = 0

            min_sequences = min(sample_sizes) if sample_sizes else 1
            max_sequences = max(sample_sizes) if sample_sizes else 1
            avg_sequences = sum(sample_sizes) / len(sample_sizes) if sample_sizes else 1

            print(f"   ⚠️  Generic variable-length data detected")

        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "sequence_length": sequence_length,
            "is_multivariate": is_multivariate,
            "data_shape": X.shape,
            "n_classes": len(np.unique(y)) if y is not None else None,
            "has_missing_values": False,  # Will be checked in statistical analysis
            "missing_ratio": 0.0,
            "is_variable_length": True,
            "variable_dim": variable_dim,
            "min_sequences": min_sequences,
            "max_sequences": max_sequences,
            "avg_sequences": avg_sequences,
            "typical_inner_shape": typical_shape
        }

    def _analyze_temporal_characteristics(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal characteristics of time series"""
        # Handle variable-length sequence data
        if X.ndim == 1 and X.dtype == object:
            return self._analyze_variable_length_temporal(X)

        if X.ndim == 2:
            # Univariate case
            sample_series = X[0]  # Take first series as representative
        else:
            # Multivariate case - use first feature of first sample
            sample_series = X[0, 0]

        # Trend analysis
        time_points = np.arange(len(sample_series))
        slope, _, r_value, _, _ = stats.linregress(time_points, sample_series)

        # Seasonality/Periodicity detection (simple approach)
        autocorr = np.correlate(sample_series, sample_series, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize

        # Find peaks in autocorrelation to detect periodicity
        peaks, _ = find_peaks(autocorr[1:], height=0.1)
        dominant_period = peaks[0] + 1 if len(peaks) > 0 else None

        # Stationarity test (simplified)
        # Split series into halves and compare means and variances
        mid = len(sample_series) // 2
        first_half = sample_series[:mid]
        second_half = sample_series[mid:]

        mean_diff = abs(np.mean(first_half) - np.mean(second_half))
        var_diff = abs(np.var(first_half) - np.var(second_half))

        return {
            "has_trend": abs(slope) > np.std(sample_series) / len(sample_series),
            "trend_strength": float(abs(slope)),
            "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
            "autocorrelation_peak": float(max(autocorr[1:10])) if len(autocorr) > 10 else 0.0,
            "dominant_period": int(dominant_period) if dominant_period else None,
            "is_likely_seasonal": dominant_period is not None and dominant_period < len(sample_series) // 4,
            "stationarity_score": 1.0 / (1.0 + mean_diff + var_diff),  # Higher = more stationary
            "r_squared": float(r_value ** 2)
        }

    def _analyze_statistical_properties(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze statistical properties of time series"""
        # Handle variable-length sequence data
        if X.ndim == 1 and X.dtype == object:
            return self._analyze_variable_length_statistical(X)

        # Flatten for overall statistics
        if X.ndim == 2:
            flat_data = X.flatten()
        else:
            flat_data = X.reshape(-1)

        # Remove NaN values
        flat_data = flat_data[~np.isnan(flat_data)]

        if len(flat_data) == 0:
            return {"error": "No valid data points"}

        return {
            "mean": float(np.mean(flat_data)),
            "std": float(np.std(flat_data)),
            "min": float(np.min(flat_data)),
            "max": float(np.max(flat_data)),
            "median": float(np.median(flat_data)),
            "skewness": float(stats.skew(flat_data)),
            "kurtosis": float(stats.kurtosis(flat_data)),
            "range": float(np.max(flat_data) - np.min(flat_data)),
            "coefficient_of_variation": float(np.std(flat_data) / np.mean(flat_data)) if np.mean(flat_data) != 0 else float('inf'),
            "percentile_25": float(np.percentile(flat_data, 25)),
            "percentile_75": float(np.percentile(flat_data, 75)),
            "iqr": float(np.percentile(flat_data, 75) - np.percentile(flat_data, 25))
        }

    def _analyze_class_distribution(self, y: np.ndarray) -> Dict[str, Any]:
        """Analyze class distribution and imbalance"""
        unique_classes, counts = np.unique(y, return_counts=True)

        # Calculate imbalance ratios
        max_count = np.max(counts)
        min_count = np.min(counts)

        # Calculate per-class imbalance ratios
        ir_per_class = {int(cls): float(max_count / count) for cls, count in zip(unique_classes, counts)}

        return {
            "n_classes": len(unique_classes),
            "class_counts": {int(cls): int(count) for cls, count in zip(unique_classes, counts)},
            "class_distribution": counts / np.sum(counts),
            "max_imbalance_ratio": float(max_count / min_count),
            "mean_imbalance_ratio": float(np.mean([max_count / count for count in counts])),
            "imbalance_ratio_per_class": ir_per_class,
            "is_balanced": float(max_count / min_count) <= 1.5,
            "is_highly_imbalanced": float(max_count / min_count) > 10.0,
            "majority_class": int(unique_classes[np.argmax(counts)]),
            "minority_class": int(unique_classes[np.argmin(counts)])
        }

    def _analyze_complexity(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze dataset complexity for hyperparameter guidance"""
        # Handle variable-length sequence data
        if X.ndim == 1 and X.dtype == object:
            return self._analyze_variable_length_complexity(X)

        if X.ndim == 2:
            # Univariate
            n_samples, seq_length = X.shape
            n_features = 1
        else:
            # Multivariate
            n_samples, n_features, seq_length = X.shape

        # Estimate complexity based on various factors
        length_complexity = "short" if seq_length < 100 else "medium" if seq_length < 500 else "long"
        feature_complexity = "simple" if n_features == 1 else "moderate" if n_features < 10 else "complex"
        sample_complexity = "small" if n_samples < 1000 else "medium" if n_samples < 10000 else "large"

        # Calculate signal-to-noise ratio estimate
        if X.ndim == 2:
            signal_power = np.mean(np.var(X, axis=1))
            noise_estimate = np.mean([np.var(np.diff(x)) for x in X[:10]])  # Use first 10 samples
        else:
            signal_power = np.mean([np.var(X[:, f, :], axis=1).mean() for f in range(n_features)])
            noise_estimate = np.mean([np.var(np.diff(X[:10, f, :], axis=1)) for f in range(n_features)])

        snr_estimate = signal_power / noise_estimate if noise_estimate > 0 else float('inf')

        return {
            "sequence_length_category": length_complexity,
            "feature_complexity_category": feature_complexity,
            "sample_size_category": sample_complexity,
            "estimated_snr": float(snr_estimate),
            "data_dimensionality": n_samples * n_features * seq_length,
            "complexity_score": self._calculate_complexity_score(n_samples, n_features, seq_length, snr_estimate)
        }

    def _calculate_complexity_score(self, n_samples: int, n_features: int, seq_length: int, snr: float) -> float:
        """Calculate overall complexity score (0-1, higher = more complex)"""
        # Normalize factors
        sample_factor = min(n_samples / 10000, 1.0)  # Normalize around 10k samples
        feature_factor = min(n_features / 50, 1.0)    # Normalize around 50 features
        length_factor = min(seq_length / 1000, 1.0)   # Normalize around 1000 length
        snr_factor = 1.0 / (1.0 + snr / 10.0)        # Lower SNR = higher complexity

        return float((sample_factor + feature_factor + length_factor + snr_factor) / 4.0)

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hyperparameter recommendations based on analysis"""
        basic_info = analysis["basic_info"]
        temporal = analysis["temporal_characteristics"]
        stats_props = analysis["statistical_properties"]
        complexity = analysis["complexity_metrics"]
        class_dist = analysis["class_distribution"]

        recommendations = {
            "model_architecture": {},
            "training_parameters": {},
            "data_preprocessing": {},
            "optimization_focus": []
        }

        # Model architecture recommendations
        if basic_info["sequence_length"] < 100:
            recommendations["model_architecture"]["suggested_model_types"] = ["CNN", "MLP", "Simple RNN"]
        elif basic_info["sequence_length"] < 500:
            recommendations["model_architecture"]["suggested_model_types"] = ["LSTM", "GRU", "CNN-LSTM"]
        else:
            recommendations["model_architecture"]["suggested_model_types"] = ["Transformer", "LSTM", "TCN"]

        # Training parameter recommendations
        if complexity["sample_size_category"] == "small":
            recommendations["training_parameters"]["batch_size_range"] = [16, 64]
            recommendations["training_parameters"]["learning_rate_range"] = [0.001, 0.01]
        else:
            recommendations["training_parameters"]["batch_size_range"] = [64, 256]
            recommendations["training_parameters"]["learning_rate_range"] = [0.0001, 0.01]

        # Preprocessing recommendations
        if stats_props["coefficient_of_variation"] > 2.0:
            recommendations["data_preprocessing"]["normalization"] = "StandardScaler or RobustScaler"
        else:
            recommendations["data_preprocessing"]["normalization"] = "MinMaxScaler"

        if temporal["has_trend"]:
            recommendations["data_preprocessing"]["detrending"] = "Consider detrending"

        # Class imbalance handling
        if class_dist and class_dist["max_imbalance_ratio"] > 3.0:
            recommendations["optimization_focus"].append("class_imbalance_handling")
            recommendations["training_parameters"]["class_weight_strategy"] = "balanced or custom weights"

        # Optimization focus
        if complexity["complexity_score"] > 0.7:
            recommendations["optimization_focus"].append("regularization")
            recommendations["optimization_focus"].append("model_selection")
        else:
            recommendations["optimization_focus"].append("performance_tuning")

        return recommendations

    def generate_llm_context(self, dataset_name: str) -> str:
        """Generate formatted context string for LLM prompts"""
        if dataset_name not in self.analysis_results:
            return f"Dataset: {dataset_name} (no detailed analysis available)"

        analysis = self.analysis_results[dataset_name]
        basic = analysis["basic_info"]
        temporal = analysis["temporal_characteristics"]
        stats = analysis["statistical_properties"]
        complexity = analysis["complexity_metrics"]
        class_dist = analysis["class_distribution"]
        recommendations = analysis["recommendations"]

        # Build context based on data type
        if basic.get('is_variable_length', False):
            # Variable-length sequence format
            context_parts = [
                f"Dataset: {dataset_name}",
                f"Type: {'Multivariate' if basic['is_multivariate'] else 'Univariate'} Variable-Length Time Series",
                f"Shape: {basic['n_samples']} samples (variable snippets per sample)"
            ]

            if basic.get('sequence_length'):
                context_parts.append(f"Snippet Structure: {basic['avg_sequences']:.1f} avg snippets × {basic['sequence_length']} timesteps × {basic['n_features']} features")
            else:
                context_parts.append(f"Snippet Structure: {basic['avg_sequences']:.1f} avg snippets × variable length × {basic['n_features']} features")

            if basic.get('min_sequences') and basic.get('max_sequences'):
                context_parts.append(f"Snippet Range: {basic['min_sequences']}-{basic['max_sequences']} snippets per sample")
        else:
            # Fixed-length format
            context_parts = [
                f"Dataset: {dataset_name}",
                f"Type: {'Multivariate' if basic['is_multivariate'] else 'Univariate'} Time Series",
                f"Shape: {basic['n_samples']} samples × {basic['n_features']} features × {basic['sequence_length']} timesteps"
            ]

        if class_dist:
            context_parts.append(f"Classes: {class_dist['n_classes']} classes")
            if class_dist["max_imbalance_ratio"] > 2.0:
                context_parts.append(f"Class Imbalance: Max ratio {class_dist['max_imbalance_ratio']:.2f} (imbalanced)")
            else:
                context_parts.append("Class Distribution: Balanced")

        # Temporal characteristics
        temporal_desc = []
        if temporal["has_trend"]:
            temporal_desc.append(f"{temporal['trend_direction']} trend")
        if temporal["is_likely_seasonal"]:
            temporal_desc.append(f"seasonal (period≈{temporal['dominant_period']})")
        if temporal["stationarity_score"] < 0.5:
            temporal_desc.append("non-stationary")
        else:
            temporal_desc.append("stationary")

        if temporal_desc:
            context_parts.append(f"Temporal: {', '.join(temporal_desc)}")

        # Statistical properties
        context_parts.append(f"Statistics: mean={stats['mean']:.3f}, std={stats['std']:.3f}, CV={stats['coefficient_of_variation']:.2f}")

        # Complexity and recommendations
        context_parts.append(f"Complexity: {complexity['complexity_score']:.2f} ({complexity['sequence_length_category']} sequences, {complexity['feature_complexity_category']} features)")

        if recommendations["optimization_focus"]:
            context_parts.append(f"Optimization Focus: {', '.join(recommendations['optimization_focus'])}")

        return " | ".join(context_parts)

    def get_hyperparameter_suggestions(self, dataset_name: str) -> Dict[str, Any]:
        """Get specific hyperparameter suggestions based on analysis"""
        if dataset_name not in self.analysis_results:
            return {}

        analysis = self.analysis_results[dataset_name]
        return analysis["recommendations"]

    def _analyze_variable_length_temporal(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal characteristics for variable-length sequence data"""
        print("🔍 Analyzing temporal characteristics of variable-length sequences...")

        # Sample first few sequences for analysis
        sample_series_list = []
        for i in range(min(5, len(X))):
            if hasattr(X[i], 'shape') and X[i].ndim >= 2:
                # Take first snippet of first sample, first feature if multivariate
                if X[i].ndim == 3:  # (num_snippets, seq_length, features)
                    sample_series_list.append(X[i][0, :, 0])  # First snippet, first feature
                elif X[i].ndim == 2:  # (num_snippets, features)
                    sample_series_list.append(X[i][0, :])  # First snippet, all features

        if not sample_series_list:
            return {
                "has_trend": False,
                "trend_strength": 0.0,
                "trend_direction": "unknown",
                "autocorrelation_peak": 0.0,
                "dominant_period": None,
                "is_likely_seasonal": False,
                "stationarity_score": 0.5,
                "r_squared": 0.0,
                "note": "Insufficient data for temporal analysis"
            }

        # Use first available series for analysis
        sample_series = sample_series_list[0]

        # Trend analysis
        time_points = np.arange(len(sample_series))
        slope, _, r_value, _, _ = stats.linregress(time_points, sample_series)

        # Simplified autocorrelation analysis
        try:
            autocorr = np.correlate(sample_series, sample_series, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize

            # Find peaks in autocorrelation
            peaks, _ = find_peaks(autocorr[1:], height=0.1)
            dominant_period = peaks[0] + 1 if len(peaks) > 0 else None
        except:
            autocorr = np.array([1.0])
            dominant_period = None

        # Stationarity test (simplified)
        mid = len(sample_series) // 2
        first_half = sample_series[:mid]
        second_half = sample_series[mid:]

        mean_diff = abs(np.mean(first_half) - np.mean(second_half))
        var_diff = abs(np.var(first_half) - np.var(second_half))

        return {
            "has_trend": abs(slope) > np.std(sample_series) / len(sample_series),
            "trend_strength": float(abs(slope)),
            "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
            "autocorrelation_peak": float(max(autocorr[1:10])) if len(autocorr) > 10 else 0.0,
            "dominant_period": int(dominant_period) if dominant_period else None,
            "is_likely_seasonal": dominant_period is not None and dominant_period < len(sample_series) // 4,
            "stationarity_score": 1.0 / (1.0 + mean_diff + var_diff),
            "r_squared": float(r_value ** 2),
            "note": "Analysis based on first snippet of variable-length sequences"
        }

    def _analyze_variable_length_statistical(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze statistical properties for variable-length sequence data"""
        print("🔍 Analyzing statistical properties of variable-length sequences...")

        # Collect data from all sequences
        all_data = []
        for i in range(min(100, len(X))):  # Limit samples for performance
            if hasattr(X[i], 'shape') and X[i].size > 0:
                if X[i].ndim >= 2:
                    # Flatten the sequences and collect data
                    flat_sample = X[i].flatten()
                    all_data.append(flat_sample)

        if not all_data:
            return {"error": "No valid data points in variable-length sequences"}

        # Combine all data
        flat_data = np.concatenate(all_data)

        # Remove NaN values
        flat_data = flat_data[~np.isnan(flat_data)]

        if len(flat_data) == 0:
            return {"error": "No valid data points after removing NaN"}

        return {
            "mean": float(np.mean(flat_data)),
            "std": float(np.std(flat_data)),
            "min": float(np.min(flat_data)),
            "max": float(np.max(flat_data)),
            "median": float(np.median(flat_data)),
            "skewness": float(stats.skew(flat_data)),
            "kurtosis": float(stats.kurtosis(flat_data)),
            "range": float(np.max(flat_data) - np.min(flat_data)),
            "coefficient_of_variation": float(np.std(flat_data) / np.mean(flat_data)) if np.mean(flat_data) != 0 else float('inf'),
            "percentile_25": float(np.percentile(flat_data, 25)),
            "percentile_75": float(np.percentile(flat_data, 75)),
            "iqr": float(np.percentile(flat_data, 75) - np.percentile(flat_data, 25)),
            "note": "Statistics from variable-length sequences (sampled)"
        }

    def _analyze_variable_length_complexity(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze complexity for variable-length sequence data"""
        print("🔍 Analyzing complexity of variable-length sequences...")

        n_samples = len(X)

        # Analyze structure from samples
        sample_shapes = []
        sequence_counts = []

        for i in range(min(20, n_samples)):
            if hasattr(X[i], 'shape'):
                sample_shapes.append(X[i].shape)
                if X[i].ndim >= 1:
                    sequence_counts.append(X[i].shape[0])

        if not sample_shapes:
            return {"error": "Cannot analyze complexity of variable-length data"}

        # Extract characteristics
        typical_shape = sample_shapes[0]

        if len(typical_shape) == 3:
            # (num_snippets, seq_length, features)
            n_features = typical_shape[2]
            seq_length = typical_shape[1]
        elif len(typical_shape) == 2:
            # (num_snippets, features)
            n_features = typical_shape[1]
            seq_length = None  # Variable
        else:
            n_features = 1
            seq_length = None

        # Calculate average sequence count
        avg_sequences = sum(sequence_counts) / len(sequence_counts) if sequence_counts else 1
        max_sequences = max(sequence_counts) if sequence_counts else 1

        # Complexity categories
        if seq_length:
            length_complexity = "short" if seq_length < 100 else "medium" if seq_length < 500 else "long"
        else:
            length_complexity = "variable"

        feature_complexity = "simple" if n_features == 1 else "moderate" if n_features < 10 else "complex"
        sample_complexity = "small" if n_samples < 1000 else "medium" if n_samples < 10000 else "large"
        sequence_complexity = "simple" if avg_sequences < 5 else "moderate" if avg_sequences < 20 else "complex"

        # Estimate total data dimensionality
        if seq_length:
            total_dimensionality = n_samples * avg_sequences * seq_length * n_features
        else:
            total_dimensionality = n_samples * avg_sequences * n_features  # Rough estimate

        # Complexity score (0-1, higher = more complex)
        sample_factor = min(n_samples / 10000, 1.0)
        feature_factor = min(n_features / 50, 1.0)
        sequence_factor = min(avg_sequences / 50, 1.0)
        length_factor = min(seq_length / 1000, 1.0) if seq_length else 0.5

        complexity_score = (sample_factor + feature_factor + sequence_factor + length_factor) / 4.0

        return {
            "sequence_length_category": length_complexity,
            "feature_complexity_category": feature_complexity,
            "sample_size_category": sample_complexity,
            "sequence_count_category": sequence_complexity,
            "estimated_snr": 1.0,  # Cannot easily estimate for variable-length
            "data_dimensionality": int(total_dimensionality),
            "complexity_score": float(complexity_score),
            "avg_sequences_per_sample": float(avg_sequences),
            "max_sequences_per_sample": int(max_sequences),
            "note": "Complexity analysis for variable-length sequences"
        }