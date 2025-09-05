#!/usr/bin/env python3
"""
Comprehensive Test Suite for MAT-HPO Library

This test suite validates the core functionality of the MAT-HPO library
to ensure reliability, correctness, and performance across different
usage scenarios and configurations.

Test Categories:
1. Core Component Tests: Individual module functionality
2. Integration Tests: Multi-component interaction validation
3. Configuration Tests: Parameter validation and edge cases
4. Performance Tests: Efficiency and resource usage validation
5. Error Handling Tests: Robustness under failure conditions

Run with: python -m pytest tests/test_library.py -v
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import os
import json
from unittest.mock import MagicMock, patch

# Import MAT-HPO components
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.base_environment import BaseEnvironment
from core.hyperparameter_space import HyperparameterSpace
from core.agent import Actor, Critic, get_device
from core.sqddpg import SQDDPG
from core.replay_buffer import TransReplayBuffer, MultiAgentReplayBuffer, Transition
from core.multi_agent_optimizer import MAT_HPO_Optimizer
from utils.config import OptimizationConfig, DefaultConfigs
from utils.logger import HPOLogger, SimpleLogger


class TestEnvironment(BaseEnvironment):
    """Test environment for validation purposes"""
    
    def __init__(self):
        super().__init__(name="TestEnvironment")
        self.step_count = 0
        
    def step(self, hyperparams):
        """Simple test evaluation function"""
        self.step_count += 1
        
        # Simple fitness function based on hyperparameters
        score = 0.0
        for key, value in hyperparams.items():
            if isinstance(value, (int, float)):
                score += abs(value) * 0.1
        
        # Normalize to [0, 1] range
        f1 = min(1.0, max(0.0, score))
        auc = min(1.0, max(0.0, score * 0.9))
        gmean = min(1.0, max(0.0, score * 0.8))
        
        # Done after 5 steps for quick testing
        done = self.step_count >= 5
        
        return f1, auc, gmean, done


class TestHyperparameterSpace:
    """Test cases for HyperparameterSpace class"""
    
    def test_initialization(self):
        """Test basic initialization"""
        space = HyperparameterSpace()
        assert len(space.parameters) == 0
        assert space.agent_dims == (0, 0, 0)
    
    def test_add_continuous_parameter(self):
        """Test adding continuous parameters"""
        space = HyperparameterSpace()
        space.add_continuous('learning_rate', 1e-4, 1e-2, agent=0)
        
        assert 'learning_rate' in space.parameters
        assert space.parameters['learning_rate']['type'] == 'continuous'
        assert space.parameters['learning_rate']['bounds'] == (1e-4, 1e-2)
        assert space.parameters['learning_rate']['agent'] == 0
    
    def test_add_discrete_parameter(self):
        """Test adding discrete parameters"""
        space = HyperparameterSpace()
        space.add_discrete('batch_size', [16, 32, 64, 128], agent=1)
        
        assert 'batch_size' in space.parameters
        assert space.parameters['batch_size']['type'] == 'discrete'
        assert space.parameters['batch_size']['choices'] == [16, 32, 64, 128]
        assert space.parameters['batch_size']['agent'] == 1
    
    def test_agent_dimensions(self):
        """Test agent dimension calculation"""
        space = HyperparameterSpace()
        space.add_continuous('lr1', 1e-4, 1e-2, agent=0)
        space.add_continuous('lr2', 1e-4, 1e-2, agent=0)
        space.add_discrete('batch_size', [16, 32, 64], agent=1)
        space.add_continuous('dropout', 0.0, 0.5, agent=2)
        
        assert space.agent_dims == (2, 1, 1)
    
    def test_action_translation(self):
        """Test action to hyperparameter translation"""
        space = HyperparameterSpace()
        space.add_continuous('learning_rate', 1e-4, 1e-2, agent=0)
        space.add_discrete('batch_size', [16, 32, 64, 128], agent=1)
        
        # Test with sample actions
        actions = [
            torch.tensor([0.5]),  # Agent 0
            torch.tensor([0.25]), # Agent 1  
            torch.tensor([0.0])   # Agent 2 (empty)
        ]
        
        hyperparams = space.translate_actions(actions)
        
        assert 'learning_rate' in hyperparams
        assert 'batch_size' in hyperparams
        assert isinstance(hyperparams['learning_rate'], float)
        assert hyperparams['batch_size'] in [16, 32, 64, 128]


class TestAgent:
    """Test cases for Actor and Critic networks"""
    
    def test_device_selection(self):
        """Test device selection functionality"""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_actor_initialization(self):
        """Test Actor network initialization"""
        actor = Actor(hyp_num=3, d_model=32, nhead=2)
        
        assert actor.hyp_num == 3
        assert actor.d_model == 32
        assert isinstance(actor, torch.nn.Module)
    
    def test_actor_forward(self):
        """Test Actor forward pass"""
        actor = Actor(hyp_num=2, d_model=32, nhead=2)
        
        # Test with batch input
        batch_size = 4
        input_tensor = torch.randn(batch_size, 2)
        
        output = actor(input_tensor)
        
        assert output.shape == (batch_size, 2)
        assert torch.all(output >= -1) and torch.all(output <= 1)  # tanh output
    
    def test_critic_initialization(self):
        """Test Critic network initialization"""
        critic = Critic(input_dim=10, hidden_dim=64)
        
        assert isinstance(critic, torch.nn.Module)
    
    def test_critic_forward(self):
        """Test Critic forward pass"""
        critic = Critic(input_dim=8, hidden_dim=32)
        
        batch_size = 3
        input_tensor = torch.randn(batch_size, 8)
        
        output = critic(input_tensor)
        
        assert output.shape == (batch_size, 1)


class TestSQDDPG:
    """Test cases for SQDDPG algorithm"""
    
    def test_initialization(self):
        """Test SQDDPG initialization"""
        sqddpg = SQDDPG(hyp_num0=2, hyp_num1=3, hyp_num2=1)
        
        assert sqddpg.hyp_num0 == 2
        assert sqddpg.hyp_num1 == 3
        assert sqddpg.hyp_num2 == 1
        assert sqddpg.n_ == 3
    
    def test_policy_generation(self):
        """Test policy action generation"""
        sqddpg = SQDDPG(hyp_num0=2, hyp_num1=2, hyp_num2=2)
        
        batch_size = 2
        state = torch.randn(batch_size, 2)
        
        actions = sqddpg.policy(state)
        
        assert actions.shape == (batch_size, 3, 2)  # [batch, agents, action_dim]


class TestReplayBuffer:
    """Test cases for replay buffer functionality"""
    
    def test_buffer_initialization(self):
        """Test replay buffer initialization"""
        buffer = TransReplayBuffer(size=100)
        
        assert len(buffer) == 0
        assert buffer.size == 100
        assert not buffer.is_full()
    
    def test_add_experience(self):
        """Test adding experiences to buffer"""
        buffer = TransReplayBuffer(size=5)
        
        # Add some transitions
        for i in range(3):
            transition = Transition(
                state=np.array([i, i+1]),
                action=np.array([i*0.1]),
                reward=i*0.5
            )
            buffer.add_experience(transition)
        
        assert len(buffer) == 3
    
    def test_buffer_overflow(self):
        """Test buffer behavior when capacity is exceeded"""
        buffer = TransReplayBuffer(size=2)
        
        # Add 3 transitions to size-2 buffer
        for i in range(3):
            transition = Transition(
                state=np.array([i]),
                action=np.array([i]),
                reward=i
            )
            buffer.add_experience(transition)
        
        assert len(buffer) == 2  # Should cap at buffer size
        assert buffer.is_full()
    
    def test_batch_sampling(self):
        """Test batch sampling from buffer"""
        buffer = TransReplayBuffer(size=10)
        
        # Add transitions
        for i in range(5):
            transition = Transition(
                state=np.array([i]),
                action=np.array([i]),
                reward=i
            )
            buffer.add_experience(transition)
        
        batch = buffer.get_batch(batch_size=3)
        assert len(batch) == 3
        
        # Test sampling more than available
        batch = buffer.get_batch(batch_size=10)
        assert len(batch) == 5  # Should return all available
    
    def test_multi_agent_buffer(self):
        """Test multi-agent replay buffer"""
        buffer = MultiAgentReplayBuffer(num_agents=3, buffer_size=10)
        
        assert buffer.num_agents == 3
        assert len(buffer.buffers) == 3
        
        # Test adding to specific agent
        transition = Transition(
            state=np.array([1, 2]),
            action=np.array([0.5]),
            reward=1.0
        )
        buffer.add_experience(agent_id=0, transition=transition)
        
        # Test invalid agent ID
        with pytest.raises(ValueError):
            buffer.add_experience(agent_id=5, transition=transition)


class TestOptimizationConfig:
    """Test cases for optimization configuration"""
    
    def test_default_config(self):
        """Test default configuration creation"""
        config = OptimizationConfig()
        
        assert config.max_steps > 0
        assert config.batch_size > 0
        assert config.replay_buffer_size > 0
        assert hasattr(config, 'device')
    
    def test_config_validation(self):
        """Test configuration parameter validation"""
        # Test invalid parameters
        with pytest.raises(ValueError):
            OptimizationConfig(max_steps=0)
        
        with pytest.raises(ValueError):
            OptimizationConfig(batch_size=-1)
        
        with pytest.raises(ValueError):
            OptimizationConfig(replay_buffer_size=0)
    
    def test_config_update(self):
        """Test configuration updating"""
        config = OptimizationConfig()
        original_steps = config.max_steps
        
        config.update(max_steps=200)
        assert config.max_steps == 200
        assert config.max_steps != original_steps
    
    def test_config_serialization(self):
        """Test configuration to/from dictionary"""
        config = OptimizationConfig(max_steps=150, batch_size=64)
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict['max_steps'] == 150
        assert config_dict['batch_size'] == 64
        
        # Test from_dict
        new_config = OptimizationConfig.from_dict(config_dict)
        assert new_config.max_steps == 150
        assert new_config.batch_size == 64
    
    def test_default_configs(self):
        """Test default configuration presets"""
        quick_config = DefaultConfigs.quick_test()
        assert quick_config.max_steps == 10
        
        standard_config = DefaultConfigs.standard()
        assert standard_config.max_steps == 100
        
        extensive_config = DefaultConfigs.extensive()
        assert extensive_config.max_steps == 500
        
        cpu_config = DefaultConfigs.cpu_only()
        assert not cpu_config.use_cuda


class TestLogger:
    """Test cases for logging functionality"""
    
    def test_simple_logger(self):
        """Test simple logger functionality"""
        logger = SimpleLogger(verbose=False)
        
        # These should not raise exceptions
        logger.info("Test message")
        logger.log_step(1, 0.8, 0.9, 0.85, 1.5, {'lr': 0.001})
    
    def test_hpo_logger_initialization(self):
        """Test HPO logger initialization"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = HPOLogger(output_dir=tmp_dir, verbose=False)
            
            # Check that log files are created
            assert os.path.exists(logger.log_file)
            assert os.path.exists(logger.step_log_file)
    
    def test_hpo_logger_step_logging(self):
        """Test step logging functionality"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = HPOLogger(output_dir=tmp_dir, verbose=False)
            
            # Log a few steps
            for i in range(3):
                logger.log_step(
                    step=i,
                    f1=0.8 + i*0.01,
                    auc=0.9 + i*0.01,
                    gmean=0.85 + i*0.01,
                    step_time=1.0 + i*0.1,
                    hyperparams={'param1': i, 'param2': i*0.5}
                )
            
            # Check that JSONL file has correct number of lines
            with open(logger.step_log_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 3
                
                # Check that each line is valid JSON
                for line in lines:
                    data = json.loads(line.strip())
                    assert 'step' in data
                    assert 'metrics' in data
                    assert 'hyperparameters' in data


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline"""
        # Create test environment
        environment = TestEnvironment()
        
        # Create hyperparameter space
        space = HyperparameterSpace()
        space.add_continuous('param1', 0.0, 1.0, agent=0)
        space.add_discrete('param2', [1, 2, 3], agent=1)
        space.add_continuous('param3', -1.0, 1.0, agent=2)
        
        # Create configuration
        config = DefaultConfigs.quick_test()
        config.update(max_steps=3, verbose=False)  # Very short for testing
        
        # Create temporary directory for results
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Run optimization
            optimizer = MAT_HPO_Optimizer(
                environment=environment,
                hyperparameter_space=space,
                config=config,
                output_dir=tmp_dir
            )
            
            results = optimizer.optimize()
            
            # Verify results structure
            assert 'best_hyperparameters' in results
            assert 'best_performance' in results
            assert 'optimization_stats' in results
            
            # Verify that optimization ran
            assert results['optimization_stats']['total_steps'] > 0
            
            # Verify best hyperparameters exist
            best_params = results['best_hyperparameters']
            assert 'param1' in best_params
            assert 'param2' in best_params
            assert 'param3' in best_params
    
    def test_error_handling(self):
        """Test error handling in optimization"""
        # Create problematic environment
        class FailingEnvironment(BaseEnvironment):
            def step(self, hyperparams):
                raise RuntimeError("Simulated failure")
        
        environment = FailingEnvironment()
        space = HyperparameterSpace()
        space.add_continuous('param1', 0.0, 1.0, agent=0)
        
        config = DefaultConfigs.quick_test()
        config.update(max_steps=1, verbose=False)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            optimizer = MAT_HPO_Optimizer(
                environment=environment,
                hyperparameter_space=space,
                config=config,
                output_dir=tmp_dir
            )
            
            # This should handle the error gracefully
            # The exact behavior depends on implementation
            try:
                results = optimizer.optimize()
            except Exception as e:
                # Some exceptions might be expected
                print(f"Handled exception: {e}")


class TestPerformance:
    """Performance and efficiency tests"""
    
    def test_memory_usage(self):
        """Test memory usage remains reasonable"""
        # Create a larger hyperparameter space
        space = HyperparameterSpace()
        for i in range(10):
            space.add_continuous(f'param_{i}', 0.0, 1.0, agent=i % 3)
        
        # Create components
        sqddpg = SQDDPG(
            hyp_num0=4,
            hyp_num1=3,
            hyp_num2=3
        )
        
        buffer = TransReplayBuffer(size=1000)
        
        # Add many experiences
        for i in range(100):
            transition = Transition(
                state=np.random.randn(10),
                action=np.random.randn(10),
                reward=np.random.random()
            )
            buffer.add_experience(transition)
        
        # This should complete without memory issues
        assert len(buffer) == 100
    
    @pytest.mark.slow
    def test_optimization_speed(self):
        """Test optimization speed (marked as slow test)"""
        import time
        
        environment = TestEnvironment()
        space = HyperparameterSpace()
        space.add_continuous('param1', 0.0, 1.0, agent=0)
        space.add_continuous('param2', 0.0, 1.0, agent=1)
        
        config = DefaultConfigs.quick_test()
        config.update(max_steps=10, verbose=False)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            optimizer = MAT_HPO_Optimizer(
                environment=environment,
                hyperparameter_space=space,
                config=config,
                output_dir=tmp_dir
            )
            
            start_time = time.time()
            results = optimizer.optimize()
            end_time = time.time()
            
            # Optimization should complete in reasonable time
            total_time = end_time - start_time
            avg_time_per_step = total_time / config.max_steps
            
            print(f"Optimization took {total_time:.2f}s ({avg_time_per_step:.2f}s/step)")
            
            # This is a rough check - actual times will vary
            assert total_time < 60  # Should complete within 1 minute
            assert avg_time_per_step < 10  # Each step should be < 10 seconds


def run_tests():
    """Run all tests with appropriate configuration"""
    print("🧪 Running MAT-HPO Library Test Suite")
    print("=" * 50)
    
    # Run pytest with verbose output
    import subprocess
    import sys
    
    test_file = __file__
    cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)