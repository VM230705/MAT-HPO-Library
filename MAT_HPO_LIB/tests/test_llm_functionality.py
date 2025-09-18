"""
Test suite for LLM functionality in MAT_HPO_LIB
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json

from MAT_HPO_LIB import (
    LLMEnhancedMAT_HPO_Optimizer,
    LLMEnhancedOptimizationConfig,
    BaseEnvironment,
    HyperparameterSpace,
    OllamaLLMClient,
    LLMHyperparameterMixer,
    LLaPipeAdaptiveAdvisor,
    PerformanceMetricCalculator,
    DatasetInfoReader,
    LLMConversationLogger
)


class MockEnvironment(BaseEnvironment):
    """Mock environment for testing"""
    
    def __init__(self):
        super().__init__()
        self.name = "MockEnvironment"
        self.dataset_name = "TestDataset"
        self.total_samples = 1000
        self.num_features = 10
        self.num_classes = 2
        self.step_count = 0
    
    def reset(self):
        return torch.zeros(5)
    
    def step(self, hyperparams):
        self.step_count += 1
        # Return reasonable mock values
        f1 = 0.7 + np.random.random() * 0.2
        auc = 0.7 + np.random.random() * 0.2  
        gmean = 0.7 + np.random.random() * 0.2
        done = self.step_count >= 10
        return f1, auc, gmean, done
    
    def get_dataset_info(self):
        return {
            'total_samples': self.total_samples,
            'num_features': self.num_features,
            'num_classes': self.num_classes
        }


def create_test_hyperparameter_space():
    """Create test hyperparameter space"""
    space = HyperparameterSpace()
    space.add_continuous('learning_rate', 1e-4, 1e-2, agent=0)
    space.add_continuous('batch_size', 16, 64, agent=1)
    space.add_discrete('optimizer', ['adam', 'sgd'], agent=2)
    return space


class TestOllamaLLMClient:
    """Test OllamaLLMClient functionality"""
    
    def test_client_initialization(self):
        """Test client can be initialized"""
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"response": "Hello"}
            
            client = OllamaLLMClient()
            assert client.model_name == "llama3.2:3b"
            assert "11434" in client.base_url
    
    @patch('requests.post')
    def test_generate_hyperparameters_success(self, mock_post):
        """Test successful hyperparameter generation"""
        # Mock successful LLM response
        mock_response = {
            "response": '{"learning_rate": 0.001, "batch_size": 32, "optimizer": "adam"}'
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        client = OllamaLLMClient()
        space = create_test_hyperparameter_space()
        
        result = client.generate_hyperparameters(
            hyperparameter_space=space,
            dataset_info={'name': 'TestDataset'},
            training_history=[]
        )
        
        assert isinstance(result, dict)
        assert 'learning_rate' in result
        assert 'batch_size' in result
        assert 'optimizer' in result
    
    @patch('requests.post')
    def test_generate_hyperparameters_failure_fallback(self, mock_post):
        """Test fallback when LLM fails"""
        # Mock failed LLM response
        mock_post.side_effect = Exception("Connection failed")
        
        client = OllamaLLMClient()
        space = create_test_hyperparameter_space()
        
        result = client.generate_hyperparameters(
            hyperparameter_space=space,
            dataset_info={'name': 'TestDataset'},
            training_history=[]
        )
        
        # Should return fallback parameters
        assert isinstance(result, dict)
        # Should have all required parameters
        if hasattr(space, 'parameters'):
            for param_name in space.parameters.keys():
                assert param_name in result


class TestLLMHyperparameterMixer:
    """Test LLMHyperparameterMixer functionality"""
    
    def test_mixer_initialization_fixed_alpha(self):
        """Test mixer initialization with fixed alpha"""
        mixer = LLMHyperparameterMixer(
            alpha=0.3,
            dataset_name="TestDataset",
            use_adaptive_trigger=False
        )
        
        assert mixer.alpha == 0.3
        assert mixer.use_adaptive_trigger == False
        assert mixer.adaptive_advisor is None
    
    def test_mixer_initialization_adaptive(self):
        """Test mixer initialization with adaptive triggering"""
        mixer = LLMHyperparameterMixer(
            alpha=0.3,
            dataset_name="TestDataset", 
            use_adaptive_trigger=True,
            slope_threshold=0.01
        )
        
        assert mixer.use_adaptive_trigger == True
        assert mixer.adaptive_advisor is not None
        assert mixer.adaptive_advisor.slope_threshold == 0.01
    
    def test_should_use_llm_fixed_alpha(self):
        """Test LLM decision with fixed alpha"""
        mixer = LLMHyperparameterMixer(
            alpha=0.5,  # 50% probability
            dataset_name="TestDataset",
            use_adaptive_trigger=False
        )
        
        # Test multiple decisions to check probability
        decisions = []
        for _ in range(100):
            use_llm, info = mixer.should_use_llm()
            decisions.append(use_llm)
        
        # Should be roughly 50% (with some random variation)
        llm_rate = sum(decisions) / len(decisions)
        assert 0.3 < llm_rate < 0.7  # Allow for randomness
        
        # Check decision info
        use_llm, info = mixer.should_use_llm()
        assert info['method'] == 'fixed_alpha'
        assert info['alpha'] == 0.5
    
    def test_update_history(self):
        """Test history update functionality"""
        mixer = LLMHyperparameterMixer(alpha=0.3, dataset_name="TestDataset")
        
        # Add some history
        metrics = {'f1': 0.8, 'auc': 0.85, 'gmean': 0.82}
        hyperparams = {'learning_rate': 0.001, 'batch_size': 32}
        
        mixer.update_history(metrics, hyperparams, step=1)
        
        assert len(mixer.training_history) == 1
        assert mixer.training_history[0]['metrics'] == metrics
        assert mixer.training_history[0]['hyperparams'] == hyperparams
        assert mixer.training_history[0]['step'] == 1
    
    def test_get_usage_stats(self):
        """Test usage statistics"""
        mixer = LLMHyperparameterMixer(alpha=0.3, dataset_name="TestDataset")
        
        # Simulate some decisions
        mixer.rl_decision_count = 7
        mixer.llm_decision_count = 3
        
        stats = mixer.get_usage_stats()
        
        assert stats['rl_count'] == 7
        assert stats['llm_count'] == 3
        assert stats['total'] == 10
        assert stats['rl_pct'] == 70.0
        assert stats['llm_pct'] == 30.0
        assert stats['adaptive_mode'] == False
        assert stats['fixed_alpha'] == 0.3


class TestLLaPipeAdaptiveAdvisor:
    """Test LLaPipeAdaptiveAdvisor functionality"""
    
    def test_advisor_initialization(self):
        """Test advisor initialization"""
        advisor = LLaPipeAdaptiveAdvisor(
            slope_threshold=0.01,
            buffer_size=10,
            cooldown_period=5,
            uncertainty_threshold=0.1
        )
        
        assert advisor.slope_threshold == 0.01
        assert advisor.buffer_size == 10
        assert advisor.cooldown_period == 5
        assert advisor.uncertainty_threshold == 0.1
        assert advisor.episode_count == 0
    
    def test_detect_uncertainty(self):
        """Test uncertainty detection"""
        advisor = LLaPipeAdaptiveAdvisor()
        
        # Test with high variance Q-values (uncertain)
        q_values = torch.tensor([0.1, 0.5, 0.9])
        uncertain, score = advisor.detect_uncertainty(q_values)
        assert uncertain == True
        assert score > 0
        
        # Test with low variance Q-values (certain)
        q_values = torch.tensor([0.5, 0.51, 0.49])
        uncertain, score = advisor.detect_uncertainty(q_values)
        assert uncertain == False or score < 0.1  # Low uncertainty
    
    def test_detect_stagnation(self):
        """Test stagnation detection"""
        advisor = LLaPipeAdaptiveAdvisor(slope_threshold=0.01)
        
        # Add improving performance (should not trigger)
        performances = [0.5, 0.55, 0.6, 0.65, 0.7]
        for perf in performances:
            stagnant, info = advisor.detect_stagnation(perf)
        
        # Should not be stagnant with improving performance
        assert stagnant == False
        assert info['slope'] > 0.01  # Positive slope above threshold
        
        # Add stagnant performance
        stagnant_performances = [0.7, 0.7, 0.7, 0.7]
        for perf in stagnant_performances:
            stagnant, info = advisor.detect_stagnation(perf)
        
        # Should detect stagnation
        assert stagnant == True
        assert info['slope'] < 0.01  # Low slope indicating stagnation
    
    def test_should_trigger_llm(self):
        """Test LLM triggering logic"""
        advisor = LLaPipeAdaptiveAdvisor(
            slope_threshold=0.01,
            min_episodes_before_trigger=3
        )
        
        # First few episodes should not trigger (insufficient episodes)
        for i in range(2):
            should_trigger, info = advisor.should_trigger_llm(0.5)
            assert should_trigger == False
            assert info['trigger_reason'] == 'insufficient_episodes'
        
        # Add stagnant performance to trigger
        stagnant_perf = [0.5, 0.5, 0.5, 0.5]
        for perf in stagnant_perf:
            should_trigger, info = advisor.should_trigger_llm(perf)
        
        # Should trigger due to stagnation
        assert should_trigger == True
        assert 'stagnation' in info['trigger_reason']
    
    def test_get_statistics(self):
        """Test statistics generation"""
        advisor = LLaPipeAdaptiveAdvisor()
        
        # Simulate some activity
        advisor.episode_count = 10
        advisor.trigger_count = 2
        advisor.successful_interventions = 1
        
        stats = advisor.get_statistics()
        
        assert stats['total_episodes'] == 10
        assert stats['total_triggers'] == 2
        assert stats['successful_interventions'] == 1
        assert stats['trigger_rate'] == 0.2  # 2/10
        assert stats['success_rate'] == 0.5   # 1/2


class TestPerformanceMetricCalculator:
    """Test PerformanceMetricCalculator functionality"""
    
    def test_calculate_unified_metric(self):
        """Test unified metric calculation"""
        # Test with valid gmean
        unified = PerformanceMetricCalculator.calculate_unified_metric(0.8, 0.7, 0.75)
        assert 0 <= unified <= 1
        
        # Test with zero gmean
        unified = PerformanceMetricCalculator.calculate_unified_metric(0.8, 0.7, 0.0)
        assert 0 <= unified <= 1
    
    def test_adaptive_metric_selection(self):
        """Test adaptive metric selection"""
        # Test with valid gmean (should return gmean)
        result = PerformanceMetricCalculator.adaptive_metric_selection(0.8, 0.7, 0.75)
        assert result == 0.75
        
        # Test with low gmean (should return f1)
        result = PerformanceMetricCalculator.adaptive_metric_selection(0.8, 0.7, 0.05)
        assert result == 0.8
        
        # Test with zero f1 (should return auc)
        result = PerformanceMetricCalculator.adaptive_metric_selection(0.0, 0.7, 0.05)
        assert result == 0.7


class TestLLMEnhancedOptimizer:
    """Test LLM-Enhanced Optimizer functionality"""
    
    def test_config_creation(self):
        """Test LLM configuration creation"""
        config = LLMEnhancedOptimizationConfig(
            max_steps=10,
            enable_llm=True,
            llm_alpha=0.3,
            use_adaptive_trigger=True,
            slope_threshold=0.01
        )
        
        assert config.max_steps == 10
        assert config.enable_llm == True
        assert config.llm_alpha == 0.3
        assert config.use_adaptive_trigger == True
        assert config.slope_threshold == 0.01
    
    def test_optimizer_initialization_with_llm_disabled(self):
        """Test optimizer initialization with LLM disabled"""
        env = MockEnvironment()
        space = create_test_hyperparameter_space()
        config = LLMEnhancedOptimizationConfig(
            max_steps=5,
            enable_llm=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            optimizer = LLMEnhancedMAT_HPO_Optimizer(env, space, config, temp_dir)
            assert optimizer.llm_mixer is None
    
    @patch('MAT_HPO_LIB.llm.llm_client.OllamaLLMClient')
    def test_optimizer_initialization_with_llm_enabled(self, mock_client):
        """Test optimizer initialization with LLM enabled"""
        env = MockEnvironment()
        space = create_test_hyperparameter_space()
        config = LLMEnhancedOptimizationConfig(
            max_steps=5,
            enable_llm=True,
            llm_alpha=0.3
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            optimizer = LLMEnhancedMAT_HPO_Optimizer(env, space, config, temp_dir)
            assert optimizer.llm_mixer is not None
            assert optimizer.llm_mixer.alpha == 0.3


class TestDatasetInfoReader:
    """Test DatasetInfoReader functionality"""
    
    def test_reader_initialization_no_csv(self):
        """Test reader initialization when CSV doesn't exist"""
        reader = DatasetInfoReader("nonexistent.csv")
        assert reader.dataset_info is None
    
    def test_get_dataset_stats_no_data(self):
        """Test getting stats when no data is available"""
        reader = DatasetInfoReader("nonexistent.csv")
        stats = reader.get_dataset_stats("TestDataset")
        assert stats is None
    
    def test_categorization_methods(self):
        """Test dataset categorization methods"""
        reader = DatasetInfoReader("nonexistent.csv")
        
        # Test dataset size categorization
        assert reader._categorize_dataset_size(100) == "VERY_SMALL"
        assert reader._categorize_dataset_size(1000) == "SMALL"
        assert reader._categorize_dataset_size(5000) == "MEDIUM"
        assert reader._categorize_dataset_size(20000) == "LARGE"
        
        # Test imbalance categorization
        assert reader._categorize_imbalance(1.2) == "BALANCED"
        assert reader._categorize_imbalance(2.5) == "MILD_IMBALANCE"
        assert reader._categorize_imbalance(5.0) == "MODERATE_IMBALANCE"
        assert reader._categorize_imbalance(15.0) == "SEVERE_IMBALANCE"
        
        # Test complexity estimation
        complexity = reader._estimate_complexity(2, 10, 100)
        assert complexity in ["LOW", "MEDIUM", "HIGH"]


class TestLLMConversationLogger:
    """Test LLMConversationLogger functionality"""
    
    def test_logger_initialization(self):
        """Test logger initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = LLMConversationLogger(
                dataset_name="TestDataset",
                mode="test_mode", 
                output_dir=temp_dir
            )
            
            assert logger.dataset_name == "TestDataset"
            assert logger.mode == "test_mode"
            assert logger.output_dir == temp_dir
            assert logger.total_llm_calls == 0
    
    def test_log_llm_conversation(self):
        """Test conversation logging"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = LLMConversationLogger(
                dataset_name="TestDataset",
                mode="test_mode",
                output_dir=temp_dir
            )
            
            logger.log_llm_conversation(
                step=1,
                attempt=1,
                prompt="Test prompt",
                response="Test response", 
                parse_success=True,
                parsed_params={"param": 0.5}
            )
            
            assert logger.total_llm_calls == 1
            assert logger.successful_parses == 1
            
            # Check that log file was created
            log_files = [f for f in os.listdir(temp_dir) if f.endswith('.jsonl')]
            assert len(log_files) > 0
    
    def test_get_statistics(self):
        """Test statistics generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = LLMConversationLogger(
                dataset_name="TestDataset",
                mode="test_mode",
                output_dir=temp_dir
            )
            
            # Simulate some activity
            logger.total_llm_calls = 5
            logger.successful_parses = 3
            logger.failed_parses = 2
            logger.total_attempts = 7
            
            stats = logger.get_statistics()
            
            assert stats['total_llm_calls'] == 5
            assert stats['successful_parses'] == 3
            assert stats['failed_parses'] == 2
            assert stats['success_rate'] == 3/7
            assert stats['avg_attempts_per_call'] == 7/5


# Integration tests
class TestLLMIntegration:
    """Integration tests for LLM functionality"""
    
    @patch('MAT_HPO_LIB.llm.llm_client.OllamaLLMClient')
    def test_end_to_end_llm_optimization(self, mock_client_class):
        """Test end-to-end optimization with mocked LLM"""
        # Mock LLM client
        mock_client = Mock()
        mock_client.generate_hyperparameters.return_value = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'adam'
        }
        mock_client_class.return_value = mock_client
        
        # Setup
        env = MockEnvironment()
        space = create_test_hyperparameter_space()
        config = LLMEnhancedOptimizationConfig(
            max_steps=3,  # Short test
            enable_llm=True,
            llm_alpha=1.0,  # Always use LLM
            batch_size=2,
            replay_buffer_size=10
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            optimizer = LLMEnhancedMAT_HPO_Optimizer(env, space, config, temp_dir)
            results = optimizer.optimize()
            
            # Check results
            assert 'best_performance' in results
            assert 'llm_statistics' in results
            assert results['llm_statistics']['llm_pct'] > 0
            
            # Check that LLM was called
            assert mock_client.generate_hyperparameters.called


def run_tests():
    """Run all tests"""
    print("🧪 Running LLM functionality tests...")
    
    # Run tests using pytest programmatically
    test_result = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])
    
    if test_result == 0:
        print("✅ All LLM tests passed!")
    else:
        print("❌ Some LLM tests failed!")
    
    return test_result == 0


if __name__ == "__main__":
    run_tests()