"""
SPNV2 Integration Example for MAT_HPO_LIB

This example demonstrates how to integrate the enhanced LLM capabilities
of MAT_HPO_LIB with the SPNV2 project for ECG classification hyperparameter
optimization.

Usage:
    python spnv2_integration_example.py --dataset ICBEB --fold 1 --steps 30
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Add MAT_HPO_LIB to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import MAT_HPO_LIB components
from MAT_HPO_LIB import (
    BaseLLMClient,
    EnhancedLLMHyperparameterMixer,
    AdaptiveAlphaController,
    OllamaLLMClient
)
from MAT_HPO_LIB.core import BaseEnvironment, HyperparameterSpace, LLMEnhancedMAT_HPO_Optimizer, LLMEnhancedOptimizationConfig

# Add SPNV2 to path (assuming it's in the same project)
project_root = Path(__file__).parent.parent.parent.parent
spnv2_path = project_root / "SPNV2"
sys.path.insert(0, str(spnv2_path))

# Import SPNV2 components
try:
    from configs.backbone_config import get_backbone_config
    from models.snippet_cnnlstm import SnippetCNNLSTM
    from utils.data_loader import get_data_loaders
    from utils.training_utils import train_epoch, evaluate_model
    SPNV2_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SPNV2 components not available: {e}")
    print("Make sure you're running from the correct directory and SPNV2 is set up properly.")
    SPNV2_AVAILABLE = False


class SPNV2Environment(BaseEnvironment):
    """
    SPNV2-specific environment for hyperparameter optimization

    Integrates with SPNV2's ECG classification pipeline to optimize
    hyperparameters for the CNN-LSTM model using LLM guidance.
    """

    def __init__(self, dataset_name: str, fold: int, device: str = 'cuda', max_epochs: int = 10):
        """
        Initialize SPNV2 environment

        Args:
            dataset_name: Dataset name (ICBEB or PTBXL)
            fold: Cross-validation fold number
            device: Device to use for training
            max_epochs: Maximum training epochs per evaluation
        """
        super().__init__()

        if not SPNV2_AVAILABLE:
            raise ImportError("SPNV2 components are not available. Check your installation.")

        self.dataset_name = dataset_name
        self.fold = fold
        self.device = device
        self.max_epochs = max_epochs

        # Load SPNV2 configuration
        self.config = get_backbone_config(dataset_name.lower())

        # Initialize data loaders
        self.train_loader, self.val_loader, self.test_loader = self._load_data()

        # Set dataset info for LLM
        self.total_samples = len(self.train_loader.dataset)
        self.num_features = self.config.input_size
        self.num_classes = self.config.num_classes

        print(f"🏥 SPNV2 Environment initialized:")
        print(f"   Dataset: {dataset_name}")
        print(f"   Fold: {fold}")
        print(f"   Samples: {self.total_samples}")
        print(f"   Classes: {self.num_classes}")
        print(f"   Max epochs: {max_epochs}")

    def _load_data(self):
        """Load SPNV2 data loaders"""
        try:
            # Use SPNV2's data loading utility
            train_loader, val_loader, test_loader = get_data_loaders(
                dataset=self.dataset_name.lower(),
                fold=self.fold,
                batch_size=32,  # Default batch size, will be optimized
                mode='hpo'  # Use HPO mode for validation split
            )
            return train_loader, val_loader, test_loader
        except Exception as e:
            print(f"❌ Failed to load SPNV2 data: {e}")
            raise

    def reset(self) -> torch.Tensor:
        """Reset environment state"""
        # Return a simple state tensor (can be enhanced with dataset characteristics)
        state = torch.tensor([
            self.total_samples / 10000.0,  # Normalized sample count
            self.num_features / 1000.0,    # Normalized feature count
            self.num_classes / 10.0,       # Normalized class count
            self.fold / 10.0               # Normalized fold number
        ], dtype=torch.float32)

        return state

    def step(self, hyperparams: Dict[str, Any]) -> Tuple[float, float, float, bool]:
        """
        Evaluate hyperparameters by training SPNV2 model

        Args:
            hyperparams: Dictionary of hyperparameters to evaluate

        Returns:
            Tuple of (f1_score, auc_score, gmean_score, done)
        """
        try:
            # Create model with hyperparameters
            model = self._create_model(hyperparams)
            model.to(self.device)

            # Setup optimizer and loss function
            optimizer = self._create_optimizer(model, hyperparams)
            criterion = self._create_loss_function(hyperparams)

            # Training loop
            best_val_acc = 0.0
            patience_counter = 0
            max_patience = 5

            for epoch in range(self.max_epochs):
                # Training phase
                model.train()
                train_loss = train_epoch(model, self.train_loader, optimizer, criterion, self.device)

                # Validation phase
                model.eval()
                val_metrics = evaluate_model(model, self.val_loader, self.device)

                val_acc = val_metrics['accuracy']

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        break

            # Final evaluation on validation set
            model.eval()
            final_metrics = evaluate_model(model, self.val_loader, self.device)

            f1_score = final_metrics['f1']
            auc_score = final_metrics.get('auc', 0.0)
            gmean_score = final_metrics.get('gmean', 0.0)

            # Cleanup
            del model
            torch.cuda.empty_cache()

            return f1_score, auc_score, gmean_score, False

        except Exception as e:
            print(f"❌ Error in SPNV2 step: {e}")
            # Return poor performance on error
            return 0.0, 0.0, 0.0, False

    def _create_model(self, hyperparams: Dict[str, Any]) -> nn.Module:
        """Create SPNV2 model with hyperparameters"""
        # Update config with hyperparameters
        model_config = self.config.copy()

        # Map hyperparameters to model config
        if 'hidden_size' in hyperparams:
            model_config.hidden_size = int(hyperparams['hidden_size'])
        if 'num_layers' in hyperparams:
            model_config.num_layers = int(hyperparams['num_layers'])
        if 'dropout' in hyperparams:
            model_config.dropout = float(hyperparams['dropout'])

        # Create model
        model = SnippetCNNLSTM(model_config)

        return model

    def _create_optimizer(self, model: nn.Module, hyperparams: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer with hyperparameters"""
        learning_rate = hyperparams.get('learning_rate', 0.001)
        weight_decay = hyperparams.get('weight_decay', 1e-5)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        return optimizer

    def _create_loss_function(self, hyperparams: Dict[str, Any]) -> nn.Module:
        """Create loss function with class weights if specified"""
        # Extract class weights from hyperparameters
        class_weights = []
        for i in range(self.num_classes):
            weight_key = f'class_weight_{i}'
            if weight_key in hyperparams:
                class_weights.append(hyperparams[weight_key])

        if class_weights:
            weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        return criterion

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information for LLM prompts"""
        return {
            'name': self.dataset_name,
            'type': 'ECG Time Series',
            'total_samples': self.total_samples,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'fold': self.fold,
            'sequence_length': self.config.input_size,
            'problem_type': 'multi_class_classification',
            'domain': 'medical_ecg'
        }


def create_spnv2_hyperparameter_space(dataset_name: str) -> HyperparameterSpace:
    """Create hyperparameter space for SPNV2 optimization"""

    # Get dataset-specific configuration
    config = get_backbone_config(dataset_name.lower())
    num_classes = config.num_classes

    # Define hyperparameter space
    parameters = {}

    # Agent 0: Loss/Weight parameters
    for i in range(num_classes):
        parameters[f'class_weight_{i}'] = {
            'type': 'continuous',
            'bounds': [0.1, 3.0],
            'agent': 0
        }

    # Agent 1: Model architecture parameters
    parameters['hidden_size'] = {
        'type': 'discrete',
        'choices': [32, 64, 128, 256],
        'agent': 1
    }

    parameters['num_layers'] = {
        'type': 'discrete',
        'choices': [1, 2, 3],
        'agent': 1
    }

    parameters['dropout'] = {
        'type': 'continuous',
        'bounds': [0.0, 0.5],
        'agent': 1
    }

    # Agent 2: Training parameters
    parameters['learning_rate'] = {
        'type': 'continuous',
        'bounds': [1e-5, 1e-2],
        'agent': 2
    }

    parameters['weight_decay'] = {
        'type': 'continuous',
        'bounds': [1e-6, 1e-3],
        'agent': 2
    }

    parameters['batch_size'] = {
        'type': 'discrete',
        'choices': [16, 32, 64, 128],
        'agent': 2
    }

    # Create hyperparameter space
    hyperparameter_space = HyperparameterSpace(
        parameters=parameters,
        agent_dims=[num_classes, 3, 3]  # Dimensions for each agent
    )

    return hyperparameter_space


def main():
    """Main function for SPNV2 LLM-enhanced HPO example"""
    parser = argparse.ArgumentParser(description='SPNV2 LLM-Enhanced HPO Example')
    parser.add_argument('--dataset', type=str, choices=['ICBEB', 'PTBXL'],
                       default='ICBEB', help='Dataset name')
    parser.add_argument('--fold', type=int, default=1, help='Cross-validation fold')
    parser.add_argument('--steps', type=int, default=30, help='Optimization steps')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--llm-strategy', type=str,
                       choices=['fixed_alpha', 'adaptive_alpha', 'llmpipe', 'hybrid'],
                       default='adaptive_alpha', help='LLM mixing strategy')
    parser.add_argument('--alpha', type=float, default=0.3, help='LLM usage probability (for fixed_alpha)')
    parser.add_argument('--slope-threshold', type=float, default=0.01,
                       help='Slope threshold for adaptive strategies')
    parser.add_argument('--max-epochs', type=int, default=10,
                       help='Maximum epochs per hyperparameter evaluation')
    parser.add_argument('--llm-client', type=str, choices=['ollama', 'openai', 'anthropic'],
                       default='ollama', help='LLM client to use')

    args = parser.parse_args()

    # Check SPNV2 availability
    if not SPNV2_AVAILABLE:
        print("❌ SPNV2 components are not available. Exiting.")
        return

    # Setup device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")

    # Create SPNV2 environment
    environment = SPNV2Environment(
        dataset_name=args.dataset,
        fold=args.fold,
        device=device,
        max_epochs=args.max_epochs
    )

    # Create hyperparameter space
    hyperparameter_space = create_spnv2_hyperparameter_space(args.dataset)

    # Create LLM client
    if args.llm_client == 'ollama':
        llm_client = OllamaLLMClient()
    elif args.llm_client == 'openai':
        try:
            from MAT_HPO_LIB.llm import OpenAILLMClient
            llm_client = OpenAILLMClient()
        except ImportError:
            print("❌ OpenAI client not available. Install with: pip install openai")
            return
    elif args.llm_client == 'anthropic':
        try:
            from MAT_HPO_LIB.llm import AnthropicLLMClient
            llm_client = AnthropicLLMClient()
        except ImportError:
            print("❌ Anthropic client not available. Install with: pip install anthropic")
            return

    # Create enhanced configuration
    config = LLMEnhancedOptimizationConfig(
        max_steps=args.steps,
        batch_size=32,
        enable_llm=True,
        llm_alpha=args.alpha,
        use_adaptive_trigger=(args.llm_strategy in ['llmpipe', 'hybrid']),
        slope_threshold=args.slope_threshold
    )

    # Create output directory
    output_dir = f"spnv2_llm_hpo_{args.dataset}_{args.fold}_{args.llm_strategy}"
    os.makedirs(output_dir, exist_ok=True)

    # Create and run optimizer
    print(f"\n🚀 Starting SPNV2 LLM-Enhanced HPO:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Fold: {args.fold}")
    print(f"   Strategy: {args.llm_strategy}")
    print(f"   Steps: {args.steps}")
    print(f"   LLM Client: {args.llm_client}")

    optimizer = LLMEnhancedMAT_HPO_Optimizer(
        environment=environment,
        hyperparameter_space=hyperparameter_space,
        config=config,
        output_dir=output_dir
    )

    # Replace the default mixer with enhanced version
    if hasattr(optimizer, 'llm_mixer'):
        optimizer.llm_mixer = EnhancedLLMHyperparameterMixer(
            llm_client=llm_client,
            alpha=args.alpha,
            dataset_name=args.dataset,
            output_dir=output_dir,
            mixing_strategy=args.llm_strategy,
            slope_threshold=args.slope_threshold
        )

    # Run optimization
    results = optimizer.optimize()

    # Print results
    print(f"\n🎯 Optimization completed!")
    print(f"   Best F1: {results['best_f1']:.4f}")
    print(f"   Best AUC: {results['best_auc']:.4f}")
    print(f"   Best G-mean: {results['best_gmean']:.4f}")
    print(f"   Results saved to: {output_dir}")

    # Save SPNV2-specific results
    spnv2_results = {
        'dataset': args.dataset,
        'fold': args.fold,
        'llm_strategy': args.llm_strategy,
        'llm_client': args.llm_client,
        'best_hyperparameters': results['best_hyperparams'],
        'best_performance': {
            'f1': results['best_f1'],
            'auc': results['best_auc'],
            'gmean': results['best_gmean']
        },
        'optimization_config': config.to_dict(),
        'llm_statistics': results.get('llm_statistics', {})
    }

    with open(f"{output_dir}/spnv2_results.json", 'w') as f:
        json.dump(spnv2_results, f, indent=2)

    print(f"📊 SPNV2-specific results saved to: {output_dir}/spnv2_results.json")


if __name__ == '__main__':
    main()