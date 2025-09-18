"""
Dataset Information Reader for MAT_HPO_LIB
Provides dataset-specific context for LLM prompts
Supports both CSV-based info and automatic time series analysis
"""

import pandas as pd
import ast
import os
import numpy as np
from typing import Dict, Optional, Union, Any
from .time_series_analyzer import TimeSeriesAnalyzer


class DatasetInfoReader:
    """
    Reads dataset information from CSV files or auto-analyzes time series data
    Provides rich context for LLM prompts with fallback to automatic analysis
    """

    def __init__(self, csv_path: str = "./Datasets_info.csv", enable_auto_analysis: bool = True):
        self.csv_path = csv_path
        self.dataset_info = None
        self.enable_auto_analysis = enable_auto_analysis
        self.time_series_analyzer = TimeSeriesAnalyzer() if enable_auto_analysis else None
        self._load_dataset_info()
    
    def _load_dataset_info(self):
        """Load dataset information from CSV file"""
        try:
            if os.path.exists(self.csv_path):
                self.dataset_info = pd.read_csv(self.csv_path)
                print(f"✅ Loaded dataset info for {len(self.dataset_info)} datasets")
            else:
                print(f"ℹ️ Dataset info CSV not found: {self.csv_path} (using basic dataset info)")
                self.dataset_info = None
        except Exception as e:
            print(f"⚠️ Error loading dataset info: {e} (using basic dataset info)")
            self.dataset_info = None
    
    def get_dataset_stats(self, dataset_name: str) -> Optional[Dict]:
        """Get comprehensive statistics for a specific dataset"""
        if self.dataset_info is None:
            return None
        
        # Find dataset row
        dataset_row = self.dataset_info[self.dataset_info['Dataset'] == dataset_name]
        
        if dataset_row.empty:
            print(f"⚠️ Dataset '{dataset_name}' not found in CSV")
            return None
        
        row = dataset_row.iloc[0]
        
        # Parse IRperLabel (string representation of dict)
        try:
            ir_per_label = ast.literal_eval(row['IRperLabel'])
        except:
            ir_per_label = {}
        
        # Create comprehensive stats dictionary
        stats = {
            'name': row['Dataset'],
            'univariate': row['Univariate'],
            'has_missing': row['Missing'],
            'equal_length': row['Equallength'],
            'total_samples': row['#Ori_Samples'],
            'num_features': row['#Features'],
            'sequence_length': row['#Sequences'],
            'num_classes': row['#Class'],
            'max_imbalance_ratio': row['MaxIR'],
            'mean_imbalance_ratio': row['MeanIR'],
            'cv_imbalance_ratio': row['CVIR'],
            'class_imbalance_ratios': ir_per_label,
            
            # Derived insights
            'dataset_size_category': self._categorize_dataset_size(row['#Ori_Samples']),
            'imbalance_severity': self._categorize_imbalance(row['MaxIR']),
            'complexity_level': self._estimate_complexity(row['#Class'], row['#Features'], row['#Sequences']),
        }
        
        return stats
    
    def _categorize_dataset_size(self, num_samples: int) -> str:
        """Categorize dataset size"""
        if num_samples < 500:
            return "VERY_SMALL"
        elif num_samples < 2000:
            return "SMALL"
        elif num_samples < 10000:
            return "MEDIUM"
        else:
            return "LARGE"
    
    def _categorize_imbalance(self, max_ir: float) -> str:
        """Categorize imbalance severity"""
        if max_ir <= 1.5:
            return "BALANCED"
        elif max_ir <= 3.0:
            return "MILD_IMBALANCE"
        elif max_ir <= 10.0:
            return "MODERATE_IMBALANCE"
        else:
            return "SEVERE_IMBALANCE"
    
    def _estimate_complexity(self, num_classes: int, num_features: int, seq_length: int) -> str:
        """Estimate problem complexity"""
        complexity_score = (num_classes * 0.3) + (num_features * 0.2) + (seq_length * 0.001)
        
        if complexity_score < 5:
            return "LOW"
        elif complexity_score < 15:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def get_optimization_recommendations(self, dataset_name: str) -> Dict:
        """Get specific optimization recommendations based on dataset characteristics"""
        stats = self.get_dataset_stats(dataset_name)
        if not stats:
            return {}
        
        recommendations = {
            'model_size': 'conservative',
            'batch_size_range': (16, 32),
            'learning_rate_range': (0.0005, 0.002),
            'class_weight_strategy': 'balanced',
            'early_stopping_patience': 15
        }
        
        # Adjust based on dataset size
        if stats['dataset_size_category'] == 'VERY_SMALL':
            recommendations.update({
                'model_size': 'very_conservative',
                'batch_size_range': (8, 16),
                'learning_rate_range': (0.0003, 0.001),
                'early_stopping_patience': 10
            })
        elif stats['dataset_size_category'] == 'LARGE':
            recommendations.update({
                'model_size': 'moderate',
                'batch_size_range': (32, 64),
                'learning_rate_range': (0.001, 0.005),
                'early_stopping_patience': 20
            })
        
        # Adjust based on imbalance
        if stats['imbalance_severity'] == 'SEVERE_IMBALANCE':
            recommendations['class_weight_strategy'] = 'aggressive'
        elif stats['imbalance_severity'] == 'BALANCED':
            recommendations['class_weight_strategy'] = 'minimal'
        
        # Adjust based on complexity
        if stats['complexity_level'] == 'HIGH':
            recommendations['model_size'] = 'moderate'
        
        return recommendations
    
    def list_available_datasets(self) -> list:
        """List all available datasets"""
        if self.dataset_info is None:
            return []
        return self.dataset_info['Dataset'].tolist()
    
    def print_dataset_summary(self, dataset_name: str):
        """Print a human-readable summary of dataset statistics"""
        stats = self.get_dataset_stats(dataset_name)
        if not stats:
            return
        
        print(f"\n📊 Dataset Summary: {dataset_name}")
        print("="*50)
        print(f"📁 Type: {'Univariate' if stats['univariate'] else 'Multivariate'} Time Series")
        print(f"📈 Samples: {stats['total_samples']:,} ({stats['dataset_size_category']})")
        print(f"🔢 Features: {stats['num_features']}")
        print(f"📏 Sequence Length: {stats['sequence_length']}")
        print(f"🏷️  Classes: {stats['num_classes']}")
        print(f"⚖️  Imbalance: {stats['max_imbalance_ratio']:.2f} ({stats['imbalance_severity']})")
        print(f"🧠 Complexity: {stats['complexity_level']}")
        
        if stats['has_missing']:
            print("⚠️  Contains missing values")
        if not stats['equal_length']:
            print("📐 Variable length sequences")


# Global instance for convenience
_dataset_reader = None


def _get_reader():
    """Get or create dataset reader instance"""
    global _dataset_reader
    if _dataset_reader is None:
        _dataset_reader = DatasetInfoReader()
    return _dataset_reader


def get_enhanced_dataset_info(dataset_name: str) -> Optional[Dict]:
    """Get enhanced dataset information for LLM prompts"""
    return _get_reader().get_dataset_stats(dataset_name)


def get_dataset_recommendations(dataset_name: str) -> Dict:
    """Get optimization recommendations for a dataset"""
    return _get_reader().get_optimization_recommendations(dataset_name)


def list_supported_datasets() -> list:
    """List all supported datasets"""
    return _get_reader().list_available_datasets()


def analyze_time_series_dataset(X: Union[np.ndarray, Any],
                               y: Optional[Union[np.ndarray, Any]] = None,
                               dataset_name: str = "Unknown") -> Dict[str, Any]:
    """
    Automatically analyze time series dataset and provide comprehensive info

    Args:
        X: Time series data (n_samples, n_features, seq_length) or (n_samples, seq_length)
        y: Labels (optional)
        dataset_name: Name of the dataset

    Returns:
        Dictionary with comprehensive dataset analysis
    """
    analyzer = TimeSeriesAnalyzer()
    return analyzer.analyze_dataset(X, y, dataset_name)


def get_time_series_llm_context(X: Union[np.ndarray, Any],
                               y: Optional[Union[np.ndarray, Any]] = None,
                               dataset_name: str = "Unknown") -> str:
    """
    Generate LLM context string from time series data analysis

    Args:
        X: Time series data
        y: Labels (optional)
        dataset_name: Name of the dataset

    Returns:
        Formatted context string for LLM prompts
    """
    analyzer = TimeSeriesAnalyzer()
    analysis = analyzer.analyze_dataset(X, y, dataset_name)
    return analyzer.generate_llm_context(dataset_name)


def get_universal_dataset_info(dataset_name: str = "Unknown",
                             X: Optional[Union[np.ndarray, Any]] = None,
                             y: Optional[Union[np.ndarray, Any]] = None) -> Dict[str, Any]:
    """
    Universal function to get dataset info - tries CSV first, falls back to auto-analysis

    Args:
        dataset_name: Name of the dataset
        X: Time series data (required if not in CSV)
        y: Labels (optional)

    Returns:
        Comprehensive dataset information
    """
    # Try CSV first
    csv_info = get_enhanced_dataset_info(dataset_name)

    if csv_info is not None:
        return {
            "source": "csv",
            "dataset_info": csv_info,
            "llm_context": _format_csv_info_for_llm(csv_info)
        }

    # Fall back to automatic analysis
    if X is not None:
        analysis = analyze_time_series_dataset(X, y, dataset_name)
        analyzer = TimeSeriesAnalyzer()
        analyzer.analysis_results[dataset_name] = analysis  # Store for context generation

        return {
            "source": "auto_analysis",
            "dataset_info": analysis,
            "llm_context": analyzer.generate_llm_context(dataset_name)
        }

    # Return minimal info if nothing available
    return {
        "source": "minimal",
        "dataset_info": {"dataset_name": dataset_name},
        "llm_context": f"Dataset: {dataset_name} (no detailed information available)"
    }


def _format_csv_info_for_llm(stats: Dict[str, Any]) -> str:
    """Format CSV-based dataset info for LLM context"""
    context_parts = [
        f"Dataset: {stats['name']}",
        f"Type: {'Univariate' if stats['univariate'] else 'Multivariate'} Time Series",
        f"Shape: {stats['total_samples']} samples × {stats['num_features']} features × {stats['sequence_length']} timesteps",
        f"Classes: {stats['num_classes']} classes"
    ]

    if stats['max_imbalance_ratio'] > 2.0:
        context_parts.append(f"Class Imbalance: Max ratio {stats['max_imbalance_ratio']:.2f} ({stats['imbalance_severity']})")
    else:
        context_parts.append("Class Distribution: Balanced")

    context_parts.extend([
        f"Dataset Size: {stats['dataset_size_category']}",
        f"Complexity: {stats['complexity_level']}"
    ])

    if stats['has_missing']:
        context_parts.append("Contains missing values")
    if not stats['equal_length']:
        context_parts.append("Variable length sequences")

    return " | ".join(context_parts)