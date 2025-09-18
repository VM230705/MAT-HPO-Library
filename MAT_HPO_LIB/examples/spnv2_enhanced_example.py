"""
Enhanced SPNV2 Integration Example using simplified MAT_HPO_LIB interface

This example shows how to use the new TimeSeriesEnvironment and automatic
dataset analysis features for ECG classification tasks.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any

# Import MAT_HPO_LIB components with new interface
from MAT_HPO_LIB import (
    TimeSeriesEnvironment,
    LLMEnhancedMAT_HPO_Optimizer,
    HyperparameterSpace, ContinuousParameter, DiscreteParameter, CategoricalParameter,
    TimeSeriesLLMConfig
)


class SPLTimeSeriesEnvironment(TimeSeriesEnvironment):
    """
    SPNV2 Environment using enhanced time series interface
    Automatically analyzes ECG data and provides LLM context
    """

    def __init__(self, dataset_name: str = "ICBEB", fold: int = 1):
        super().__init__(
            name=f"SPL-{dataset_name}-Fold{fold}",
            dataset_name=dataset_name,
            validation_split=0.1,
            enable_auto_analysis=True
        )

        self.fold = fold
        self.num_classes = 9 if dataset_name.upper() == 'ICBEB' else 5

        # Data storage
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_labels = None
        self.val_labels = None
        self.test_labels = None

    def load_data(self):
        """Load ECG data and perform automatic analysis"""
        # In real implementation, load actual data
        # For demo, create synthetic ECG-like data
        n_samples = 1000
        seq_length = 1000

        # Generate synthetic ECG data
        X = np.random.randn(n_samples, seq_length)
        y = np.random.randint(0, self.num_classes, n_samples)

        # Set dataset for automatic analysis
        self.set_dataset(X, y, self.dataset_name)

        # Split data (simplified version)
        n_train = int(0.8 * n_samples)
        n_val = int(0.1 * n_samples)

        self.train_data = X[:n_train]
        self.train_labels = y[:n_train]
        self.val_data = X[n_train:n_train + n_val]
        self.val_labels = y[n_train:n_train + n_val]
        self.test_data = X[n_train + n_val:]
        self.test_labels = y[n_train + n_val:]

        print(f"✅ Data loaded: train={len(self.train_data)}, val={len(self.val_data)}, test={len(self.test_data)}")
        print(f"🤖 LLM Context: {self.get_llm_context()}")

        return {
            'X_train': self.train_data, 'y_train': self.train_labels,
            'X_val': self.val_data, 'y_val': self.val_labels,
            'X_test': self.test_data, 'y_test': self.test_labels
        }

    def create_model(self, hyperparams: Dict[str, Any]):
        """Create CNN-LSTM model with given hyperparameters"""

        class CNNLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
                super().__init__()
                self.cnn = nn.Conv1d(1, 64, kernel_size=3, padding=1)
                self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, dropout=dropout)
                self.classifier = nn.Linear(hidden_size, num_classes)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                # x: (batch, seq_len)
                x = x.unsqueeze(1)  # Add channel dimension
                x = torch.relu(self.cnn(x))  # (batch, 64, seq_len)
                x = x.transpose(1, 2)  # (batch, seq_len, 64)
                lstm_out, _ = self.lstm(x)
                x = self.dropout(lstm_out[:, -1, :])  # Use last output
                return self.classifier(x)

        model = CNNLSTM(
            input_size=1,
            hidden_size=int(hyperparams.get('hidden_size', 64)),
            num_layers=int(hyperparams.get('num_layers', 2)),
            num_classes=self.num_classes,
            dropout=float(hyperparams.get('dropout', 0.2))
        )

        return model

    def train_evaluate(self, model, hyperparams: Dict[str, Any]) -> Dict[str, float]:
        """Train and evaluate model (simplified implementation)"""

        # Get training parameters
        batch_size = int(hyperparams.get('batch_size', 32))
        learning_rate = float(hyperparams.get('learning_rate', 0.001))
        epochs = 10  # Simplified for demo

        # Create data loaders (simplified)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Use class weights if recommended by analysis
        if self.should_use_class_weights():
            class_weights = self.get_suggested_class_weights()
            if class_weights:
                weights = torch.tensor([class_weights.get(i, 1.0) for i in range(self.num_classes)])
                criterion = nn.CrossEntropyLoss(weight=weights.to(device))
                print(f"📊 Using class weights: {class_weights}")

        # Training loop (simplified)
        model.train()
        for epoch in range(epochs):
            # Simplified batch processing
            n_batches = len(self.train_data) // batch_size
            for i in range(0, len(self.train_data), batch_size):
                batch_data = torch.tensor(self.train_data[i:i+batch_size], dtype=torch.float32).to(device)
                batch_labels = torch.tensor(self.train_labels[i:i+batch_size], dtype=torch.long).to(device)

                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            # Validation
            val_data_tensor = torch.tensor(self.val_data, dtype=torch.float32).to(device)
            val_outputs = model(val_data_tensor)
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_acc = np.mean(val_preds == self.val_labels)

            # Test
            test_data_tensor = torch.tensor(self.test_data, dtype=torch.float32).to(device)
            test_outputs = model(test_data_tensor)
            test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
            test_acc = np.mean(test_preds == self.test_labels)

            # Calculate F1 (simplified)
            from sklearn.metrics import f1_score
            test_f1 = f1_score(self.test_labels, test_preds, average='weighted')

        return {
            'val_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'f1': float(test_f1),
            'auc': float(test_acc * 0.9),  # Simplified AUC estimate
            'gmean': float(np.sqrt(test_acc * test_f1))  # Simplified G-mean
        }

    def compute_reward(self, metrics: Dict[str, float]) -> float:
        """Compute reward focusing on validation accuracy and F1"""
        val_acc = metrics.get('val_accuracy', 0.0)
        f1 = metrics.get('f1', 0.0)

        # Weighted combination emphasizing validation performance
        reward = 0.7 * val_acc + 0.3 * f1
        return reward


def create_spl_hyperparameter_space() -> HyperparameterSpace:
    """Create hyperparameter space for SPL optimization"""

    space = HyperparameterSpace()

    # Model architecture parameters (Agent 1)
    space.add_parameter("hidden_size", DiscreteParameter([32, 64, 128, 256]))
    space.add_parameter("num_layers", DiscreteParameter([1, 2, 3]))
    space.add_parameter("dropout", ContinuousParameter(0.1, 0.5))

    # Training parameters (Agent 2)
    space.add_parameter("batch_size", DiscreteParameter([16, 32, 64, 128]))
    space.add_parameter("learning_rate", ContinuousParameter(0.0001, 0.01, log_scale=True))

    # Class weights (Agent 0) - will be added dynamically based on dataset analysis
    # This is handled automatically by the environment

    return space


def run_spl_hpo_example():
    """Run SPL HPO example with enhanced time series interface"""

    print("🚀 Starting Enhanced SPL HPO Example")
    print("="*50)

    # Create environment with automatic dataset analysis
    environment = SPLTimeSeriesEnvironment(dataset_name="ICBEB", fold=1)

    # Load data (this triggers automatic analysis)
    environment.load_data()

    # Create hyperparameter space
    hyperparameter_space = create_spl_hyperparameter_space()

    # Create LLM-enhanced configuration
    config = TimeSeriesLLMConfig.for_ecg_classification(
        dataset_name="ICBEB",
        n_classes=9,
        sequence_length=1000,
        enable_llm=True
    )

    # Override some settings for quick demo
    config.optimization.n_steps = 5  # Quick demo
    config.llm.min_episodes_before_llm = 2
    config.llm.llm_cooldown_episodes = 2

    print(f"📊 Dataset Complexity: {environment.get_complexity_metrics()}")
    print(f"💡 Recommendations: {environment.get_dataset_recommendations()}")

    # Create LLM-enhanced optimizer
    optimizer = LLMEnhancedMAT_HPO_Optimizer(
        environment=environment,
        hyperparameter_space=hyperparameter_space,
        config=config.optimization,
        # LLM specific configuration
        enable_llm=config.llm.enable_llm,
        llm_model=config.llm.llm_client_config.model_name,
        llm_base_url=config.llm.llm_client_config.base_url,
        mixing_strategy=config.llm.llm_strategy,
        alpha=config.llm.mixing_alpha,
        slope_threshold=config.llm.slope_threshold,
        llm_cooldown_episodes=config.llm.llm_cooldown_episodes,
        min_episodes_before_llm=config.llm.min_episodes_before_llm,
        dataset_name=config.dataset.dataset_name,
        # Pass LLM context from automatic analysis
        task_description=environment.get_llm_context()
    )

    # Run optimization
    print("\n🔄 Starting LLM-Enhanced Optimization...")
    results = optimizer.optimize()

    print("\n✅ Optimization Complete!")
    print(f"🏆 Best Performance: {results['best_reward']:.4f}")
    print(f"📊 Best Hyperparameters: {results['best_hyperparameters']}")

    return results


if __name__ == "__main__":
    # Run the example
    results = run_spl_hpo_example()

    print("\n📈 Final Results Summary:")
    print(f"Best Reward: {results['best_reward']:.4f}")
    print(f"Total Steps: {results['optimization_steps']}")
    print("Best Hyperparameters:")
    for param, value in results['best_hyperparameters'].items():
        print(f"  {param}: {value}")