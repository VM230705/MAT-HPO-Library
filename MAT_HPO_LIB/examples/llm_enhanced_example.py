"""
Example: LLM-Enhanced Hyperparameter Optimization with MAT_HPO_LIB

This example demonstrates how to use the LLM-enhanced optimizer for 
improved hyperparameter search using Large Language Models.
"""

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from MAT_HPO_LIB import (
    LLMEnhancedMAT_HPO_Optimizer, 
    LLMEnhancedOptimizationConfig,
    BaseEnvironment, 
    HyperparameterSpace
)


class RandomForestEnvironment(BaseEnvironment):
    """Example environment for Random Forest hyperparameter optimization"""
    
    def __init__(self, dataset_name="SyntheticDataset"):
        super().__init__()
        self.name = "RandomForestEnvironment"
        self.dataset_name = dataset_name
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare synthetic dataset"""
        # Generate synthetic classification dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=1,
            class_sep=0.7,
            random_state=42
        )
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Store dataset info for LLM
        self.total_samples = len(X)
        self.num_features = X.shape[1]
        self.num_classes = len(np.unique(y))
        
        print(f"📊 Dataset prepared: {self.total_samples} samples, {self.num_features} features, {self.num_classes} classes")
    
    def get_dataset_info(self):
        """Provide dataset information for LLM prompts"""
        return {
            'total_samples': self.total_samples,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'task_type': 'classification',
            'is_synthetic': True
        }
    
    def reset(self):
        """Reset environment state"""
        return torch.zeros(10)  # Simple state representation
    
    def step(self, hyperparams):
        """
        Evaluate hyperparameters by training RandomForest
        
        Args:
            hyperparams: Dictionary of hyperparameter values
            
        Returns:
            f1, auc, gmean, done
        """
        try:
            # Extract hyperparameters
            n_estimators = int(hyperparams.get('n_estimators', 100))
            max_depth = int(hyperparams.get('max_depth', 10))
            min_samples_split = int(hyperparams.get('min_samples_split', 2))
            min_samples_leaf = int(hyperparams.get('min_samples_leaf', 1))
            max_features = hyperparams.get('max_features', 'sqrt')
            
            # Train Random Forest
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=1
            )
            
            model.fit(self.X_train, self.y_train)
            
            # Evaluate
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Calculate G-mean (geometric mean of sensitivity and specificity)
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            gmean = np.sqrt(sensitivity * specificity)
            
            # Simple termination condition
            done = f1 > 0.95
            
            return f1, auc, gmean, done
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return 0.0, 0.5, 0.0, False


def create_hyperparameter_space():
    """Define hyperparameter search space"""
    space = HyperparameterSpace()
    
    # Agent 0: Tree structure parameters
    space.add_discrete('max_features', ['sqrt', 'log2', None], agent=0)
    
    # Agent 1: Tree complexity parameters  
    space.add_continuous('max_depth', 3, 20, agent=1)
    space.add_continuous('min_samples_split', 2, 10, agent=1)
    space.add_continuous('min_samples_leaf', 1, 5, agent=1)
    
    # Agent 2: Ensemble parameters
    space.add_continuous('n_estimators', 10, 200, agent=2)
    
    return space


def run_pure_rl_example():
    """Example with pure RL optimization (no LLM)"""
    print("🚀 Running Pure RL Example")
    print("="*50)
    
    # Setup environment and space
    env = RandomForestEnvironment("SyntheticDataset_RL")
    space = create_hyperparameter_space()
    
    # Configure pure RL optimization  
    config = LLMEnhancedOptimizationConfig(
        max_steps=10,  # Reduced for faster testing
        enable_llm=False,  # Disable LLM
        batch_size=4,
        replay_buffer_size=50,
        policy_learning_rate=1e-4,
        value_learning_rate=1e-3,
        verbose=True,
        early_stop_patience=5  # Early stopping for faster completion
    )
    
    # Run optimization
    optimizer = LLMEnhancedMAT_HPO_Optimizer(env, space, config)
    results = optimizer.optimize()
    
    print(f"\n🎯 Pure RL Results:")
    print(f"   Best F1: {results['best_performance']['f1']:.4f}")
    print(f"   Best AUC: {results['best_performance']['auc']:.4f}")
    print(f"   Best G-mean: {results['best_performance']['gmean']:.4f}")
    print(f"   Best hyperparams: {results['best_hyperparameters']}")
    
    return results


def run_llm_fixed_alpha_example():
    """Example with fixed alpha LLM mixing"""
    print("\n🤖 Running LLM Fixed Alpha Example")
    print("="*50)
    
    # Setup environment and space
    env = RandomForestEnvironment("SyntheticDataset_LLM_Fixed")
    space = create_hyperparameter_space()
    
    # Configure LLM with fixed alpha
    config = LLMEnhancedOptimizationConfig(
        max_steps=10,  # Reduced for faster testing
        enable_llm=True,
        llm_alpha=0.3,  # 30% chance of using LLM
        use_adaptive_trigger=False,
        batch_size=4,
        replay_buffer_size=50,
        policy_learning_rate=1e-4,
        value_learning_rate=1e-3,
        verbose=True,
        early_stop_patience=5
    )
    
    # Run optimization
    optimizer = LLMEnhancedMAT_HPO_Optimizer(env, space, config)
    results = optimizer.optimize()
    
    print(f"\n🎯 LLM Fixed Alpha Results:")
    print(f"   Best F1: {results['best_performance']['f1']:.4f}")
    print(f"   Best AUC: {results['best_performance']['auc']:.4f}")
    print(f"   Best G-mean: {results['best_performance']['gmean']:.4f}")
    print(f"   LLM usage: {results['llm_statistics']['llm_pct']:.1f}%")
    print(f"   Best hyperparams: {results['best_hyperparameters']}")
    
    return results


def run_llm_adaptive_example():
    """Example with adaptive LLM triggering"""
    print("\n🎯 Running LLM Adaptive Example")
    print("="*50)
    
    # Setup environment and space
    env = RandomForestEnvironment("SyntheticDataset_LLM_Adaptive")
    space = create_hyperparameter_space()
    
    # Configure LLM with adaptive triggering
    config = LLMEnhancedOptimizationConfig(
        max_steps=12,  # Reduced for faster testing
        enable_llm=True,
        use_adaptive_trigger=True,
        slope_threshold=0.005,  # Trigger when learning slope < 0.005
        batch_size=4,
        replay_buffer_size=50,
        policy_learning_rate=1e-4,
        value_learning_rate=1e-3,
        verbose=True,
        early_stop_patience=5
    )
    
    # Run optimization
    optimizer = LLMEnhancedMAT_HPO_Optimizer(env, space, config)
    results = optimizer.optimize()
    
    print(f"\n🎯 LLM Adaptive Results:")
    print(f"   Best F1: {results['best_performance']['f1']:.4f}")
    print(f"   Best AUC: {results['best_performance']['auc']:.4f}")
    print(f"   Best G-mean: {results['best_performance']['gmean']:.4f}")
    llm_stats = results['llm_statistics']
    if 'trigger_rate' in llm_stats:
        print(f"   LLM trigger rate: {llm_stats['trigger_rate']:.1%}")
    print(f"   LLM usage: {llm_stats['llm_pct']:.1f}%")
    print(f"   Best hyperparams: {results['best_hyperparameters']}")
    
    return results


def main():
    """Run all examples and compare results"""
    print("🧪 MAT_HPO_LIB LLM Integration Examples")
    print("="*60)
    
    results = {}
    
    # Run different optimization strategies
    try:
        results['pure_rl'] = run_pure_rl_example()
    except Exception as e:
        print(f"❌ Pure RL example failed: {e}")
        results['pure_rl'] = None
    
    try:
        results['llm_fixed'] = run_llm_fixed_alpha_example()
    except Exception as e:
        print(f"❌ LLM fixed alpha example failed: {e}")
        results['llm_fixed'] = None
    
    try:
        results['llm_adaptive'] = run_llm_adaptive_example()
    except Exception as e:
        print(f"❌ LLM adaptive example failed: {e}")
        results['llm_adaptive'] = None
    
    # Compare results
    print("\n📊 COMPARISON OF OPTIMIZATION STRATEGIES")
    print("="*60)
    
    for strategy, result in results.items():
        if result is not None:
            perf = result['best_performance']
            print(f"{strategy.upper():15} | F1: {perf['f1']:.4f} | AUC: {perf['auc']:.4f} | G-mean: {perf['gmean']:.4f}")
            
            # Show LLM stats if available
            if 'llm_statistics' in result and result['llm_statistics'].get('enabled', True):
                llm_stats = result['llm_statistics']
                print(f"{'':15} | LLM usage: {llm_stats.get('llm_pct', 0):.1f}% | Total decisions: {llm_stats.get('total', 0)}")
        else:
            print(f"{strategy.upper():15} | FAILED")
    
    print("\n✅ Examples completed!")
    print("\nNOTE: For LLM examples to work properly, make sure:")
    print("1. Ollama is running: 'ollama serve'")
    print("2. Model is available: 'ollama pull llama3.2:3b'")
    print("3. Check the output directories for detailed logs")


if __name__ == "__main__":
    main()