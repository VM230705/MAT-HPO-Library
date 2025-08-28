#!/usr/bin/env python3
"""
Simple MAT-HPO Example: Hyperparameter Optimization for a Basic ML Model

This example demonstrates how to use the MAT-HPO library to optimize hyperparameters
for a simple machine learning model. It serves as an introduction to the library's
core concepts and usage patterns.

The example:
1. Creates a simple synthetic classification dataset
2. Defines a basic hyperparameter optimization environment
3. Configures the hyperparameter search space
4. Runs the multi-agent optimization
5. Analyzes and displays the results

This is ideal for:
- Learning the MAT-HPO library basics
- Quick prototyping and testing
- Understanding the optimization workflow
- Validating library installation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Import MAT-HPO components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.base_environment import BaseEnvironment
from core.hyperparameter_space import HyperparameterSpace
from core.multi_agent_optimizer import MAT_HPO_Optimizer
from utils.config import OptimizationConfig, DefaultConfigs


class SimpleMLModel(nn.Module):
    """Simple feedforward neural network for classification"""
    
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)


class SimpleOptimizationEnvironment(BaseEnvironment):
    """
    Simple environment for hyperparameter optimization of a basic ML model.
    
    This environment evaluates hyperparameters by:
    1. Creating and training a neural network with the given hyperparameters
    2. Evaluating the trained model on validation data
    3. Returning performance metrics (F1, AUC, G-mean)
    """
    
    def __init__(self, X_train, X_val, y_train, y_val, max_epochs=50):
        super().__init__(name="SimpleMLOptimization")
        
        # Convert to tensors
        self.X_train = torch.FloatTensor(X_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_train = torch.LongTensor(y_train)
        self.y_val = torch.LongTensor(y_val)
        
        self.input_dim = X_train.shape[1]
        self.max_epochs = max_epochs
        
        print(f"📊 Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val samples")
        print(f"📐 Features: {self.input_dim}")
    
    def step(self, hyperparams):
        """
        Evaluate hyperparameters by training and testing a model.
        
        Args:
            hyperparams: Dictionary containing hyperparameter values
            
        Returns:
            f1_score, auc_score, gmean_score, done
        """
        try:
            # Extract hyperparameters
            hidden_size = int(hyperparams.get('hidden_size', 64))
            learning_rate = float(hyperparams.get('learning_rate', 0.001))
            batch_size = int(hyperparams.get('batch_size', 32))
            
            # Create model
            model = SimpleMLModel(self.input_dim, hidden_size)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Create data loader for training
            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            
            # Training loop
            model.train()
            for epoch in range(self.max_epochs):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Early stopping based on loss
                if epoch_loss / len(train_loader) < 0.01:
                    break
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(self.X_val)
                val_probs = val_outputs[:, 1].numpy()  # Probability of positive class
                val_preds = torch.argmax(val_outputs, dim=1).numpy()
            
            # Calculate metrics
            f1 = f1_score(self.y_val.numpy(), val_preds)
            auc = roc_auc_score(self.y_val.numpy(), val_probs)
            
            # Calculate G-mean (geometric mean of sensitivity and specificity)
            tn = np.sum((self.y_val.numpy() == 0) & (val_preds == 0))
            fp = np.sum((self.y_val.numpy() == 0) & (val_preds == 1))
            fn = np.sum((self.y_val.numpy() == 1) & (val_preds == 0))
            tp = np.sum((self.y_val.numpy() == 1) & (val_preds == 1))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            gmean = np.sqrt(sensitivity * specificity)
            
            return f1, auc, gmean, False  # done=False to continue optimization
            
        except Exception as e:
            print(f"⚠️  Error during evaluation: {e}")
            # Return poor performance for failed evaluations
            return 0.1, 0.5, 0.1, False


def create_sample_dataset(n_samples=1000, n_features=20, n_classes=2):
    """Create a synthetic classification dataset for demonstration"""
    print("🔧 Creating synthetic classification dataset...")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features//2,
        n_redundant=n_features//4,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val


def setup_hyperparameter_space():
    """Define the hyperparameter search space for optimization"""
    print("🎯 Setting up hyperparameter search space...")
    
    space = HyperparameterSpace()
    
    # Agent 0: Model architecture parameters
    space.add_discrete('hidden_size', [32, 64, 128, 256], agent=0)
    
    # Agent 1: Training parameters  
    space.add_continuous('learning_rate', 1e-4, 1e-2, agent=1)
    space.add_discrete('batch_size', [16, 32, 64, 128], agent=1)
    
    # Agent 2: Additional optimization parameters (for demonstration)
    # Note: These aren't used in the simple model but show how to add more parameters
    space.add_continuous('l2_regularization', 1e-6, 1e-3, agent=2)
    space.add_discrete('optimizer_type', ['adam', 'sgd'], agent=2)
    
    return space


def main():
    """Main function demonstrating simple MAT-HPO usage"""
    
    print("🚀 MAT-HPO Simple Example: Hyperparameter Optimization")
    print("=" * 60)
    
    # Step 1: Create dataset
    X_train, X_val, y_train, y_val = create_sample_dataset(
        n_samples=1000, 
        n_features=20
    )
    
    # Step 2: Create optimization environment
    print("\n🏗️  Setting up optimization environment...")
    environment = SimpleOptimizationEnvironment(
        X_train, X_val, y_train, y_val,
        max_epochs=20  # Reduced for faster demonstration
    )
    
    # Step 3: Define hyperparameter search space
    hyperparameter_space = setup_hyperparameter_space()
    
    # Step 4: Configure optimization
    print("\n⚙️  Configuring optimization settings...")
    config = DefaultConfigs.quick_test()  # Use quick test config for demo
    config.update(
        max_steps=15,  # Small number for demonstration
        verbose=True,
        early_stop_patience=5
    )
    
    # Step 5: Create and run optimizer
    print("\n🎯 Starting MAT-HPO optimization...")
    print("-" * 40)
    
    optimizer = MAT_HPO_Optimizer(
        environment=environment,
        hyperparameter_space=hyperparameter_space,
        config=config,
        output_dir="./simple_hpo_results"
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Step 6: Display results
    print("\n" + "=" * 60)
    print("🎉 Optimization Results")
    print("=" * 60)
    
    best_params = results['best_hyperparameters']
    best_perf = results['best_performance']
    
    print(f"\n🏆 Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    print(f"\n📊 Best Performance:")
    print(f"   F1-Score: {best_perf['f1']:.4f}")
    print(f"   AUC: {best_perf['auc']:.4f}")
    print(f"   G-mean: {best_perf['gmean']:.4f}")
    
    print(f"\n⏱️  Optimization Stats:")
    stats = results['optimization_stats']
    print(f"   Total Steps: {stats['total_steps']}")
    print(f"   Total Time: {stats['total_time']:.2f} seconds")
    print(f"   Avg Time/Step: {stats['avg_time_per_step']:.2f} seconds")
    
    print(f"\n💾 Results saved to: ./simple_hpo_results/")
    
    # Step 7: Test best configuration
    print("\n🧪 Testing best configuration...")
    test_env = SimpleOptimizationEnvironment(X_train, X_val, y_train, y_val, max_epochs=50)
    f1, auc, gmean, _ = test_env.step(best_params)
    
    print(f"   Final Test Results:")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   AUC: {auc:.4f}") 
    print(f"   G-mean: {gmean:.4f}")
    
    print("\n✅ Simple MAT-HPO example completed successfully!")
    print("📖 Check the DETAILED_USAGE_GUIDE.md for more advanced examples.")


if __name__ == "__main__":
    main()