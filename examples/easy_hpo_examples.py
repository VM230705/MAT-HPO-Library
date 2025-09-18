#!/usr/bin/env python3
"""
EasyHPO Examples - Comprehensive examples showing different usage patterns

This file demonstrates various ways to use the EasyHPO interface for
hyperparameter optimization with minimal setup.
"""

import sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add MAT_HPO_LIB to path (adjust path as needed)
sys.path.insert(0, '/home/vm230705/MAT-HPO-Library')

def example_1_simplest_usage():
    """Example 1: Simplest possible usage - just pass data"""
    print("=" * 60)
    print("Example 1: Simplest Usage")
    print("=" * 60)

    from MAT_HPO_LIB import EasyHPO

    # Create sample data
    X, y = make_classification(n_samples=500, n_features=10, n_classes=3,
                             n_informative=8, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val samples, {len(np.unique(y))} classes")

    # One-liner optimization
    optimizer = EasyHPO(task_type="classification", max_trials=5, verbose=True)
    results = optimizer.optimize(X_train, y_train, X_val, y_val)

    print(f"\\nBest F1 Score: {results['best_performance']['f1']:.4f}")
    print(f"Best hyperparameters: {results['best_hyperparameters']}")

def example_2_quick_functions():
    """Example 2: Using quick convenience functions"""
    print("\\n" + "=" * 60)
    print("Example 2: Quick Convenience Functions")
    print("=" * 60)

    from MAT_HPO_LIB.easy_hpo import quick_optimize

    # Create sample data
    X, y = make_classification(n_samples=300, n_features=8, n_classes=2,
                             n_informative=6, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val samples")

    # Ultra-simple one-liner
    best_params = quick_optimize(
        X_train, y_train, X_val, y_val,
        task_type="classification",
        max_trials=5,
        llm_enabled=False,  # Disable LLM for faster execution
        verbose=True
    )

    print(f"\\nBest hyperparameters: {best_params}")

def example_3_time_series():
    """Example 3: Time series classification with auto-detection"""
    print("\\n" + "=" * 60)
    print("Example 3: Time Series Classification")
    print("=" * 60)

    from MAT_HPO_LIB import EasyHPO

    # Create synthetic time series data
    np.random.seed(42)
    n_samples, seq_length, n_features = 400, 50, 1
    n_classes = 2

    # Generate time series with patterns
    X = np.random.randn(n_samples, seq_length, n_features)
    for i in range(n_samples):
        if i % 2 == 0:  # Class 0: sine pattern
            X[i, :, 0] += np.sin(np.linspace(0, 2*np.pi, seq_length))
        else:  # Class 1: cosine pattern
            X[i, :, 0] += np.cos(np.linspace(0, 2*np.pi, seq_length))

    y = np.array([i % 2 for i in range(n_samples)])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Time series dataset: {X_train.shape}")

    # Time series optimization with automatic model creation
    optimizer = EasyHPO(
        task_type="time_series_classification",
        max_trials=5,
        llm_enabled=False,  # Disable LLM for faster execution
        verbose=True
    )

    results = optimizer.optimize(X_train, y_train, X_val, y_val)

    print(f"\\nBest F1 Score: {results['best_performance']['f1']:.4f}")
    print(f"Optimized LSTM hidden size: {results['best_hyperparameters'].get('hidden_size', 'N/A')}")

def example_4_llm_enhanced():
    """Example 4: LLM-enhanced optimization"""
    print("\\n" + "=" * 60)
    print("Example 4: LLM-Enhanced Optimization")
    print("=" * 60)

    from MAT_HPO_LIB import EasyHPO

    # Create sample data
    X, y = make_classification(n_samples=400, n_features=15, n_classes=4,
                             n_informative=12, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val samples, {len(np.unique(y))} classes")

    # LLM-enhanced optimization
    optimizer = EasyHPO(
        task_type="time_series_classification",
        llm_enabled=True,              # Enable LLM
        llm_model="llama3.2:3b",       # Specify LLM model
        llm_strategy="adaptive",       # Adaptive mixing strategy
        max_trials=8,                  # More trials for LLM to be effective
        verbose=True
    )

    print("\\n🤖 Starting LLM-enhanced optimization...")
    results = optimizer.optimize(X_train, y_train, X_val, y_val)

    print(f"\\n🏆 Best F1 Score: {results['best_performance']['f1']:.4f}")
    print(f"Best hyperparameters: {results['best_hyperparameters']}")

def example_5_custom_hyperparameter_space():
    """Example 5: Custom hyperparameter space definition"""
    print("\\n" + "=" * 60)
    print("Example 5: Custom Hyperparameter Space")
    print("=" * 60)

    from MAT_HPO_LIB import EasyHPO, HyperparameterSpace

    # Create sample data
    X, y = make_classification(n_samples=300, n_features=12, n_classes=2,
                             n_informative=10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val samples")

    # Define custom hyperparameter space
    custom_space = HyperparameterSpace()
    custom_space.add_continuous('learning_rate', 0.001, 0.1, agent=0)
    custom_space.add_discrete('batch_size', [8, 16, 32, 64], agent=1)
    custom_space.add_continuous('dropout', 0.0, 0.6, agent=2)
    custom_space.add_discrete('hidden_size', [32, 64, 128], agent=0)

    print("\\n🔧 Using custom hyperparameter space:")
    print(f"   Learning rate: [0.001, 0.1]")
    print(f"   Batch size: [8, 16, 32, 64]")
    print(f"   Dropout: [0.0, 0.6]")
    print(f"   Hidden size: [32, 64, 128]")

    # Optimize with custom space
    optimizer = EasyHPO(task_type="classification", max_trials=6, verbose=True)
    results = optimizer.optimize(X_train, y_train, X_val, y_val,
                               custom_space=custom_space)

    print(f"\\nBest F1 Score: {results['best_performance']['f1']:.4f}")
    print(f"Best configuration: {results['best_hyperparameters']}")

def example_6_advanced_pipeline():
    """Example 6: Advanced usage with custom pipeline components"""
    print("\\n" + "=" * 60)
    print("Example 6: Advanced Custom Pipeline")
    print("=" * 60)

    from MAT_HPO_LIB import EasyHPO
    import torch
    import torch.nn as nn

    # Create sample data
    X, y = make_classification(n_samples=500, n_features=20, n_classes=3,
                             n_informative=15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val samples")

    # Custom model factory
    def custom_model_factory(**hyperparams):
        """Create a custom neural network"""
        hidden_size = hyperparams.get('hidden_size', 64)
        dropout_rate = hyperparams.get('dropout', 0.2)
        n_layers = hyperparams.get('n_layers', 2)

        layers = []
        input_size = X_train.shape[1]

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_size, len(np.unique(y))))

        return nn.Sequential(*layers)

    # Custom trainer
    def custom_trainer(model, X_train, y_train, **hyperparams):
        """Custom training logic"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Convert data
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.LongTensor(y_train).to(device)

        # Training parameters
        lr = hyperparams.get('learning_rate', 0.001)
        batch_size = hyperparams.get('batch_size', 32)
        epochs = hyperparams.get('epochs', 10)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        model.train()
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return model

    # Custom evaluator
    def custom_evaluator(model, X_val, y_val):
        """Custom evaluation logic"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        X_tensor = torch.FloatTensor(X_val).to(device)
        y_tensor = torch.LongTensor(y_val).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y_tensor).float().mean().item()

        # Simple F1 calculation (you could use sklearn for more precise calculation)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_val, predictions.cpu().numpy(), average='weighted')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'val_accuracy': accuracy  # For EasyHPO reward calculation
        }

    # Data loader function
    def data_loader():
        return X_train, y_train, X_val, y_val

    print("\\n🔧 Using custom pipeline with:")
    print("   - Custom neural network architecture")
    print("   - Custom training loop with PyTorch")
    print("   - Custom evaluation metrics")

    # Use advanced pipeline
    optimizer = EasyHPO(
        task_type="classification",
        max_trials=5,
        llm_enabled=False,  # Disable LLM for faster execution
        verbose=True
    )

    results = optimizer.optimize_with_pipeline(
        data_loader=data_loader,
        model_factory=custom_model_factory,
        trainer=custom_trainer,
        evaluator=custom_evaluator
    )

    print(f"\\nBest F1 Score: {results['best_performance']['f1']:.4f}")
    print(f"Best configuration: {results['best_hyperparameters']}")

    # Create final model with best hyperparameters
    best_config = optimizer.get_best_config()
    final_model = custom_model_factory(**best_config)
    print(f"\\n🎯 Final model created with optimized hyperparameters")

def main():
    """Run all examples"""
    print("EasyHPO Comprehensive Examples")
    print("=" * 60)
    print("This script demonstrates various usage patterns of the EasyHPO interface.")
    print("Each example builds on the previous ones, showing increasing complexity.")
    print()

    try:
        # Basic examples
        example_1_simplest_usage()
        example_2_quick_functions()
        example_3_time_series()

        # Advanced examples
        print("\\n" + "🚨" * 20)
        print("The following examples may take longer and require LLM/GPU resources:")
        print("🚨" * 20)

        example_4_llm_enhanced()
        example_5_custom_hyperparameter_space()
        example_6_advanced_pipeline()

        print("\\n" + "🎉" * 20)
        print("All examples completed successfully!")
        print("🎉" * 20)

    except KeyboardInterrupt:
        print("\\n❌ Examples interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()