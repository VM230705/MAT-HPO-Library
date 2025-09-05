#!/usr/bin/env python3
"""
Simple functionality test for MAT-HPO Library
Test basic imports and functionality without requiring pytest
"""
import sys
import os
import tempfile
import traceback

# Add current directory to path
sys.path.insert(0, '.')

def test_imports():
    """Test that all core imports work"""
    print("🧪 Testing imports...")
    try:
        from MAT_HPO_LIB import MAT_HPO_Optimizer, BaseEnvironment, HyperparameterSpace
        from MAT_HPO_LIB.utils import OptimizationConfig, DefaultConfigs
        print("✅ Core imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_hyperparameter_space():
    """Test hyperparameter space functionality"""
    print("🧪 Testing HyperparameterSpace...")
    try:
        from MAT_HPO_LIB import HyperparameterSpace
        
        space = HyperparameterSpace()
        space.add_continuous('learning_rate', 1e-4, 1e-2, agent=0)
        space.add_discrete('batch_size', [16, 32, 64, 128], agent=1)
        space.add_continuous('dropout', 0.0, 0.5, agent=2)
        
        assert space.agent_dims == (1, 1, 1)
        assert len(space.parameters) == 3
        
        print("✅ HyperparameterSpace tests passed")
        return True
    except Exception as e:
        print(f"❌ HyperparameterSpace test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration functionality"""
    print("🧪 Testing Configuration...")
    try:
        from MAT_HPO_LIB.utils import OptimizationConfig, DefaultConfigs
        
        # Test default configs
        quick_config = DefaultConfigs.quick_test()
        standard_config = DefaultConfigs.standard()
        
        assert quick_config.max_steps == 10
        assert standard_config.max_steps == 100
        
        # Test custom config
        config = OptimizationConfig(max_steps=50, batch_size=32)
        assert config.max_steps == 50
        assert config.batch_size == 32
        
        print("✅ Configuration tests passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

class SimpleTestEnvironment:
    """Simple test environment"""
    def __init__(self, name="SimpleTest"):
        self.name = name
        self.step_count = 0
        
    def step(self, hyperparams):
        """Simple evaluation function"""
        self.step_count += 1
        
        # Simple scoring based on hyperparameters
        score = 0.0
        for key, value in hyperparams.items():
            if isinstance(value, (int, float)):
                score += abs(value) * 0.1
        
        f1 = min(1.0, max(0.0, score))
        auc = min(1.0, max(0.0, score * 0.9))
        gmean = min(1.0, max(0.0, score * 0.8))
        
        done = self.step_count >= 3  # Quick test
        return f1, auc, gmean, done

def test_environment():
    """Test basic environment functionality"""
    print("🧪 Testing Environment...")
    try:
        env = SimpleTestEnvironment()
        
        # Test step
        hyperparams = {'learning_rate': 0.001, 'batch_size': 32}
        f1, auc, gmean, done = env.step(hyperparams)
        
        assert isinstance(f1, float)
        assert isinstance(auc, float) 
        assert isinstance(gmean, float)
        assert isinstance(done, bool)
        
        print("✅ Environment tests passed")
        return True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        traceback.print_exc()
        return False

def test_basic_integration():
    """Test basic integration without full optimization"""
    print("🧪 Testing Basic Integration...")
    try:
        from MAT_HPO_LIB import HyperparameterSpace
        from MAT_HPO_LIB.utils import DefaultConfigs
        
        # Create components
        space = HyperparameterSpace()
        space.add_continuous('param1', 0.0, 1.0, agent=0)
        space.add_discrete('param2', [1, 2, 3], agent=1)
        
        config = DefaultConfigs.quick_test()
        env = SimpleTestEnvironment()
        
        # Test action translation
        import torch
        actions = [
            torch.tensor([0.5]),   # Agent 0: 1 param
            torch.tensor([0.33]),  # Agent 1: 1 param
            torch.tensor([])       # Agent 2: 0 params (empty)
        ]
        
        hyperparams = space.translate_actions(actions)
        assert 'param1' in hyperparams
        assert 'param2' in hyperparams
        
        # Test environment step
        f1, auc, gmean, done = env.step(hyperparams)
        
        print("✅ Basic Integration tests passed")
        return True
    except Exception as e:
        print(f"❌ Basic Integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 MAT-HPO Library Simple Functionality Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_hyperparameter_space,
        test_configuration,
        test_environment,
        test_basic_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All functionality tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)