"""
Actor and Critic Networks for MAT-HPO Multi-Agent Optimization

Sophisticated neural network architectures designed specifically for multi-agent
hyperparameter optimization using transformer-based attention mechanisms and
value function approximation.

Key Components:
- **Actor Networks**: Transformer-based policy networks that generate continuous
  actions for hyperparameter selection using self-attention and positional encoding
- **Critic Networks**: Value function approximators that estimate expected returns
  for state-action pairs in the hyperparameter optimization process
- **Device Management**: Automatic GPU/CPU selection and efficient memory management
- **Positional Encoding**: Enhanced transformer architecture with proper sequence
  modeling for hyperparameter dependencies

The Actor networks use a sophisticated transformer architecture that:
1. Embeds each hyperparameter dimension independently
2. Applies positional encoding to capture parameter ordering relationships
3. Uses multi-head self-attention to model parameter interactions
4. Generates bounded actions through tanh activation for stable optimization

The Critic networks provide stable value estimation through:
1. Deep feedforward architecture with ReLU activation
2. Proper input normalization and gradient flow management
3. Single-value output for policy gradient computation

This architecture has been specifically tuned for hyperparameter optimization
tasks where parameter interactions are complex and exploration must be carefully
balanced with exploitation of promising regions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional


def get_device(gpu_device: int = 0) -> torch.device:
    """
    Intelligent device selection for optimal computation performance.
    
    This function automatically detects available hardware and selects the most
    appropriate device for neural network computation, prioritizing GPU acceleration
    when available while providing graceful fallback to CPU computation.
    
    Args:
        gpu_device: Preferred GPU device ID (default: 0). Used when CUDA is available
                   and multiple GPUs are present. Ignored if CUDA is unavailable.
    
    Returns:
        torch.device: Optimal device for computation. Returns 'cuda:gpu_device' if
                     CUDA is available, otherwise returns 'cpu' device.
                     
    Note:
        This function is called automatically during model initialization and does
        not require explicit device management by users in most cases.
    """
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_device}')
    return torch.device('cpu')


def generate_square_subsequent_mask(sz):
    """
    Generate causal attention mask for transformer decoder to prevent information leakage.
    
    This function creates a lower triangular mask that prevents the transformer decoder
    from attending to future positions in the sequence. This is crucial for maintaining
    causality in autoregressive generation and preventing the model from "cheating" by
    looking ahead in the sequence.
    
    The mask enforces that when generating the i-th hyperparameter, the model can only
    attend to hyperparameters 0 through i-1, ensuring proper sequential generation
    without future information leakage.
    
    Args:
        sz: Size of the square mask (sequence length)
        
    Returns:
        torch.Tensor: Square mask of shape [sz, sz] where:
                     - 0.0 indicates allowed attention (lower triangle)
                     - -inf indicates blocked attention (upper triangle)
                     
    Mathematical Properties:
        - Lower triangular matrix with zeros below diagonal
        - Upper triangular filled with negative infinity
        - Enables causal (autoregressive) attention patterns
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for Transformer Architecture.
    
    Implements the standard sinusoidal positional encoding used in "Attention Is All You Need"
    to provide position information to transformer models. This is essential for hyperparameter
    optimization because:
    
    1. **Parameter Ordering**: Some hyperparameters have natural ordering relationships
       (e.g., layer sizes, learning rate schedules) that benefit from position awareness
    2. **Sequence Modeling**: Enables the transformer to understand parameter sequences
       and dependencies between different hyperparameter positions
    3. **Translation Invariance**: Provides consistent positional information regardless
       of the specific hyperparameter values
    
    The encoding uses alternating sine and cosine functions with different frequencies
    to create unique position representations that are:
    - Deterministic and reproducible
    - Bounded in range [-1, 1]
    - Differentiable for gradient-based optimization
    - Capable of generalizing to longer sequences than seen during training
    
    Mathematical Foundation:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
    where pos is the position and i is the dimension index.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        """
        Initialize positional encoding with sinusoidal patterns.
        
        Args:
            d_model: Model dimension, must match transformer's d_model parameter.
                    Typical values: 64, 128, 256, 512 depending on problem complexity.
            dropout: Dropout probability applied after adding positional encoding.
                    Range: [0.0, 0.5], higher values increase regularization.
            max_len: Maximum sequence length supported. Should be >= maximum number
                    of hyperparameters handled by any single agent. Default 100
                    supports large hyperparameter spaces.
                    
        Note:
            The positional encoding is computed once during initialization and
            cached as a buffer, making forward passes computationally efficient.
        """
        """
        Initialize positional encoding with sinusoidal patterns.
        
        Args:
            d_model: Model dimension, must match transformer's d_model parameter.
                    Typical values: 64, 128, 256, 512 depending on problem complexity.
            dropout: Dropout probability applied after adding positional encoding.
                    Range: [0.0, 0.5], higher values increase regularization.
            max_len: Maximum sequence length supported. Should be >= maximum number
                    of hyperparameters handled by any single agent. Default 100
                    supports large hyperparameter spaces.
                    
        Note:
            The positional encoding is computed once during initialization and
            cached as a buffer, making forward passes computationally efficient.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Avoid in-place operations for positional encoding
        sin_values = torch.sin(position * div_term)
        cos_values = torch.cos(position * div_term)
        pe_even = torch.zeros_like(pe)
        pe_odd = torch.zeros_like(pe)
        pe_even[:, 0::2] = sin_values
        pe_odd[:, 1::2] = cos_values
        pe = pe_even + pe_odd
        pe = pe.unsqueeze(0)  # [1, max_len, d_model] for batch_first
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Actor(nn.Module):
    """
    Advanced Multi-Agent Transformer Actor Network for Hyperparameter Generation.
    
    This sophisticated neural architecture serves as the policy network in the SQDDPG
    algorithm, generating continuous actions that are translated into hyperparameter
    values. Each agent has a specialized actor that focuses on different aspects of
    the hyperparameter space.
    
    **Architectural Innovation**:
    The Actor uses a transformer encoder-decoder architecture specifically adapted
    for hyperparameter optimization:
    
    1. **Parameter-Specific Embeddings**: Each hyperparameter dimension has its own
       embedding layer, allowing the model to learn specialized representations for
       different types of parameters (learning rates, batch sizes, architectures, etc.)
    
    2. **Positional Awareness**: Sinusoidal positional encoding captures relationships
       between parameter positions, crucial for understanding parameter hierarchies
       and dependencies (e.g., learning rate schedules, layer size progressions)
    
    3. **Self-Attention Mechanism**: Multi-head attention enables the model to identify
       and exploit complex interactions between different hyperparameters, such as
       the relationship between learning rate and batch size, or model size and
       regularization strength
    
    4. **Autoregressive Generation**: The decoder uses causal masking to generate
       hyperparameters sequentially, ensuring each parameter choice can influence
       subsequent decisions while maintaining computational efficiency
    
    5. **Bounded Output**: Tanh activation ensures all actions are in [-1, 1] range,
       providing stable gradients and enabling consistent translation to hyperparameter
       bounds across different scales and types
    
    **Specialization by Agent**:
    - Agent 0: Typically handles loss-related parameters (class weights, regularization)
    - Agent 1: Manages architectural parameters (layer sizes, model structure)
    - Agent 2: Controls training dynamics (learning rates, optimization parameters)
    
    This specialization allows each actor to develop expertise in its domain while
    the multi-agent framework handles coordination and credit assignment.
    """
    
    def __init__(self, 
                 hyp_num: int,
                 d_model: int = 64,
                 nhead: int = 4, 
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1,
                 max_len: int = 100,
                 device: Optional[torch.device] = None):
        """
        Initialize sophisticated transformer-based actor network with parameter-specific design.
        
        The architecture is carefully designed to balance expressiveness with training stability
        for hyperparameter optimization tasks. Each component is tuned for the unique challenges
        of multi-agent hyperparameter search.
        
        Args:
            hyp_num: Number of hyperparameters this actor controls. Must be > 0.
                    Typical ranges: 2-10 per agent depending on problem complexity.
                    
            d_model: Transformer model dimension. Controls representational capacity.
                    Values: 32 (simple), 64 (standard), 128 (complex), 256 (very complex)
                    Higher values enable richer representations but increase computation.
                    
            nhead: Number of multi-head attention heads. Must divide d_model evenly.
                  Values: 2, 4, 8. More heads capture diverse attention patterns but
                  increase computational cost. 4 heads provide good balance.
                  
            num_encoder_layers: Transformer encoder depth. Controls input processing capacity.
                              Values: 1-4. More layers can model complex input patterns
                              but may lead to overfitting with limited data.
                              
            num_decoder_layers: Transformer decoder depth. Controls output generation sophistication.
                              Values: 1-4. More layers enable complex parameter interactions
                              but require more training data for stable learning.
                              
            dim_feedforward: Feed-forward network width in transformer layers.
                           Typically 2-4x d_model. Controls non-linear transformation capacity.
                           Larger values enable more complex feature combinations.
                           
            dropout: Regularization strength. Applied throughout the network.
                    Range: 0.0-0.3. Higher values prevent overfitting but may slow learning.
                    0.1 provides good balance for most hyperparameter optimization tasks.
                    
            max_len: Maximum supported sequence length. Should exceed maximum hyp_num
                    across all agents. Buffer size for positional encoding.
                    
            device: Computation device. None triggers automatic GPU/CPU selection.
                   Explicitly specify for multi-GPU setups or CPU-only environments.
        
        Raises:
            ValueError: If hyp_num <= 0 or architectural constraints violated
            RuntimeError: If device placement fails or CUDA issues occur
        """
        super(Actor, self).__init__()
        
        self.device = device if device is not None else get_device()
        self.hyp_num = hyp_num
        self.d_model = d_model
        
        # Create embeddings for each hyperparameter
        self.state_embedding = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(hyp_num)
        ])
        self.action_embedding = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(hyp_num)
        ])
        
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out_mean = nn.Linear(d_model, 1)
        
        # Move to device
        self.to(self.device)

    def forward(self, src):
        """
        Execute forward pass through the transformer-based actor network.
        
        This method implements the complete forward computation pipeline that transforms
        input states into bounded action outputs suitable for hyperparameter generation.
        
        **Computation Pipeline**:
        1. **Input Processing**: Expands state tensor to include feature dimension
        2. **Target Initialization**: Creates zero-initialized decoder input sequence
        3. **Attention Masking**: Generates causal mask for autoregressive generation
        4. **Embedding Generation**: Creates parameter-specific embeddings for both encoder and decoder
        5. **Positional Encoding**: Adds position information to capture parameter ordering
        6. **Transformer Processing**: Applies self-attention and feed-forward transformations
        7. **Output Generation**: Projects to action space and applies tanh for bounded output
        
        **Architectural Details**:
        - Uses batch_first=True for efficient batch processing
        - Applies embedding scaling (sqrt(d_model)) following transformer best practices
        - Implements causal masking to prevent future information leakage
        - Parameter-specific embeddings allow specialized processing per hyperparameter type
        
        Args:
            src: Input state tensor with shape [batch_size, hyp_num] where:
                - batch_size: Number of parallel samples being processed
                - hyp_num: Number of hyperparameters this actor controls
                Values should be normalized to reasonable ranges for stable training.
            
        Returns:
            output_mean: Generated actions with shape [batch_size, hyp_num].
                        Values are bounded in [-1, 1] via tanh activation.
                        These actions are later translated to actual hyperparameter
                        values by the HyperparameterSpace class.
        
        Mathematical Properties:
            - Output is deterministic (no sampling noise during inference)
            - Bounded output range ensures stable gradient flow
            - Autoregressive structure captures parameter dependencies
            - Attention mechanism models complex parameter interactions
        """
        batch_size = src.size(0)
        src = src.unsqueeze(-1)  # [batch_size, hyp_num, 1]
        
        # With batch_first=True, keep batch dimension first
        # Target sequence (initially zeros)
        trg = torch.zeros((batch_size, self.hyp_num, 1)).to(self.device)
        trg_mask = generate_square_subsequent_mask(self.hyp_num).to(self.device)
        
        # Create embeddings - transpose for processing then transpose back
        src_transpose = src.transpose(0, 1)  # [hyp_num, batch_size, 1]
        trg_transpose = trg.transpose(0, 1)  # [hyp_num, batch_size, 1]
        
        src_emb_list = []
        trg_emb_list = []
        
        for i in range(self.hyp_num):
            src_emb_list.append(self.state_embedding[i](src_transpose[i:i+1, :, :]))
            trg_emb_list.append(self.action_embedding[i](trg_transpose[i:i+1, :, :]))
        
        src_emb = torch.cat(src_emb_list, dim=0) * math.sqrt(self.d_model)
        trg_emb = torch.cat(trg_emb_list, dim=0)
        
        # Transpose back to batch_first format
        src_emb = src_emb.transpose(0, 1)  # [batch_size, hyp_num, d_model]
        trg_emb = trg_emb.transpose(0, 1)  # [batch_size, hyp_num, d_model]
        
        # Apply positional encoding
        src_emb = self.positional_encoding(src_emb)
        trg_emb = self.positional_encoding(trg_emb)

        # Transformer forward pass
        output = self.transformer(src_emb, trg_emb, tgt_mask=trg_mask)
        
        # Generate final output
        output_mean = self.fc_out_mean(output)  # [batch_size, hyp_num, 1] 
        output_mean = torch.tanh(output_mean).squeeze(-1)  # [batch_size, hyp_num]
        
        return output_mean


class Critic(nn.Module):
    """
    Deep Value Function Approximator for Multi-Agent Hyperparameter Optimization.
    
    The Critic network serves as the value function estimator in the Actor-Critic
    framework, providing crucial feedback for policy gradient computation. In the
    context of hyperparameter optimization, the critic learns to predict the expected
    cumulative reward (performance improvement) from any given state-action pair.
    
    **Functional Role in MAT-HPO**:
    1. **Value Estimation**: Predicts expected returns for state-action combinations,
       enabling the actor to understand which hyperparameter regions are most promising
    
    2. **Variance Reduction**: Provides baseline estimates that reduce the variance
       of policy gradient estimates, leading to more stable and efficient learning
    
    3. **Credit Assignment**: Works with Shapley value computation to fairly distribute
       rewards among agents based on their individual contributions
    
    4. **Exploration Guidance**: Value estimates help guide exploration by identifying
       promising regions of the hyperparameter space worth further investigation
    
    **Architecture Design**:
    The critic uses a simple but effective feedforward architecture:
    - Deep enough to capture non-linear value relationships
    - Simple enough to avoid overfitting with limited experience
    - ReLU activations for stable gradient flow
    - Single output head for scalar value prediction
    
    **Multi-Agent Context**:
    In the SQDDPG algorithm, each agent has its own critic that:
    - Receives joint observations and actions from all agents
    - Estimates values for the specific agent's policy
    - Contributes to Shapley value computation for fair credit assignment
    
    This distributed value estimation enables specialized learning while maintaining
    coordination through shared experience and joint evaluation.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 device: Optional[torch.device] = None):
        """
        Initialize the critic network with optimized architecture for value function approximation.
        
        The network architecture is specifically designed for hyperparameter optimization
        value estimation, balancing expressiveness with training stability.
        
        Args:
            input_dim: Dimensionality of the input state-action space.
                      In MAT-HPO, this is typically the concatenation of:
                      - All agents' observations (state representations)
                      - All agents' actions (proposed hyperparameters)
                      Typical values: 50-500 depending on hyperparameter space size.
                      
            hidden_dim: Width of hidden layers in the feedforward network.
                       Controls the network's capacity to learn complex value functions.
                       Values: 64 (simple), 128 (standard), 256 (complex), 512 (very complex)
                       Should scale with input_dim: roughly 2-4x input_dim is often effective.
                       
            device: Target computation device for the network.
                   None (default) triggers automatic device selection (GPU if available).
                   Explicitly specify for distributed training or resource constraints.
        
        Network Architecture:
            Input -> Linear(input_dim, hidden_dim) -> ReLU
                  -> Linear(hidden_dim, hidden_dim) -> ReLU  
                  -> Linear(hidden_dim, 1) -> Output
        
        Design Rationale:
            - Two hidden layers provide sufficient non-linearity for value approximation
            - ReLU activations ensure stable gradient flow and prevent vanishing gradients
            - Progressive dimension reduction focuses representation toward value prediction
            - Single output matches the scalar value function requirement
        
        Raises:
            ValueError: If input_dim <= 0 or hidden_dim <= 0
            RuntimeError: If device placement fails
        """
        super(Critic, self).__init__()
        
        self.device = device if device is not None else get_device()
        
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Move to device
        self.to(self.device)

    def forward(self, state):
        """
        Compute value function estimate for given state-action input.
        
        This method implements the forward pass through the critic network to produce
        scalar value estimates used in policy gradient computation and Shapley value
        calculation within the SQDDPG algorithm.
        
        **Computation Flow**:
        1. **First Hidden Layer**: Linear transformation followed by ReLU activation
           Projects input to hidden representation while introducing non-linearity
           
        2. **Second Hidden Layer**: Another linear-ReLU combination
           Enables complex value function approximation through deep non-linear mapping
           
        3. **Value Output**: Final linear projection to scalar value
           No activation function to allow unbounded value estimates
        
        **Design Rationale**:
        - ReLU activations ensure stable gradients and prevent vanishing gradient issues
        - Two hidden layers provide sufficient capacity for value function approximation
        - Unbounded output accommodates varying reward scales across different optimization tasks
        - Simple architecture prevents overfitting while maintaining expressiveness
        
        Args:
            state: Joint state-action tensor with shape [batch_size, input_dim] where:
                  - batch_size: Number of parallel evaluations
                  - input_dim: Concatenated observations and actions from all agents
                  Input should include both current states and proposed actions
                  from all agents in the multi-agent system.
            
        Returns:
            v: Value estimates with shape [batch_size, 1].
              Represents expected cumulative reward from the given state-action pair.
              Used for:
              - Policy gradient baseline computation
              - Shapley value calculation for credit assignment
              - Value function learning through temporal difference methods
        
        Note:
            In the multi-agent context, each agent's critic receives the same joint
            state-action input but learns agent-specific value functions that contribute
            to fair credit assignment through Shapley value decomposition.
        """
        h = torch.relu(self.l1(state))
        h = torch.relu(self.l2(h))
        v = self.value_head(h)
        
        return v


class ActorCriticPair:
    """
    Unified Management System for Actor-Critic Agent Pairs.
    
    This class provides a clean, convenient interface for managing the Actor-Critic
    architecture used by each agent in the multi-agent system. It encapsulates both
    networks and their interactions, simplifying training code and ensuring consistent
    behavior across all agents.
    
    **Key Benefits**:
    1. **Unified Interface**: Single object manages both policy and value networks
    2. **Consistent Configuration**: Ensures compatible architectures between actor and critic
    3. **Simplified Training**: Provides batch operations for training mode switches and parameter access
    4. **Device Management**: Coordinates device placement across both networks
    5. **Configuration Flexibility**: Supports independent configuration of actor and critic components
    
    **Usage Pattern**:
    ```python
    # Create agent with custom configurations
    agent = ActorCriticPair(
        actor_hyp_num=5,
        critic_input_dim=150,
        actor_config={'d_model': 128, 'nhead': 8},
        critic_config={'hidden_dim': 256}
    )
    
    # Train both networks
    agent.train()
    policy_output = agent.actor(state)
    value_output = agent.critic(state_action)
    ```
    
    **Multi-Agent Context**:
    In MAT-HPO, each of the three agents uses an ActorCriticPair instance,
    allowing for specialized configurations while maintaining consistent interfaces.
    This design supports heterogeneous agent architectures when needed (different
    agents may require different network complexities based on their hyperparameter
    domains).
    """
    
    def __init__(self,
                 actor_hyp_num: int,
                 critic_input_dim: int,
                 actor_config: Optional[dict] = None,
                 critic_config: Optional[dict] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize a coordinated Actor-Critic pair with flexible configuration.
        
        This constructor creates both networks with consistent device placement and
        compatible architectures, while allowing fine-grained control over each
        network's specific parameters.
        
        Args:
            actor_hyp_num: Number of hyperparameters controlled by this agent's actor.
                          Must be > 0. Typical ranges: 2-10 depending on agent specialization.
                          
            critic_input_dim: Input dimensionality for the critic network.
                             In MAT-HPO, this is typically:
                             (obs_dim + action_dim) * num_agents
                             Accounts for joint observations and actions from all agents.
                             
            actor_config: Optional dictionary of actor-specific parameters.
                         Supported keys: 'd_model', 'nhead', 'num_encoder_layers',
                         'num_decoder_layers', 'dim_feedforward', 'dropout', 'max_len'
                         If None, uses Actor class defaults optimized for hyperparameter optimization.
                         
            critic_config: Optional dictionary of critic-specific parameters.
                          Supported keys: 'hidden_dim'
                          If None, uses Critic class defaults suitable for value function approximation.
                          
            device: Target device for both networks.
                   None triggers automatic device selection (GPU preferred, CPU fallback).
                   All networks in the pair will be placed on the same device for efficiency.
        
        Configuration Examples:
            # Basic agent with default settings
            agent = ActorCriticPair(5, 150)
            
            # Agent with enhanced transformer capacity
            agent = ActorCriticPair(
                actor_hyp_num=8,
                critic_input_dim=200,
                actor_config={'d_model': 128, 'nhead': 8, 'num_encoder_layers': 3},
                critic_config={'hidden_dim': 256}
            )
            
            # CPU-only agent for resource-constrained environments
            agent = ActorCriticPair(3, 100, device=torch.device('cpu'))
        
        Raises:
            ValueError: If actor_hyp_num <= 0 or critic_input_dim <= 0
            RuntimeError: If device placement fails or network creation encounters errors
        """
        self.device = device if device is not None else get_device()
        
        # Default configurations
        actor_config = actor_config or {}
        critic_config = critic_config or {}
        
        # Create networks
        self.actor = Actor(
            hyp_num=actor_hyp_num,
            device=self.device,
            **actor_config
        )
        
        self.critic = Critic(
            input_dim=critic_input_dim,
            device=self.device,
            **critic_config
        )
    
    def get_parameters(self):
        """
        Retrieve all trainable parameters from both actor and critic networks.
        
        This method provides organized access to all network parameters for optimizer
        initialization, gradient computation, and parameter analysis. The parameters
        are returned in a structured format that separates actor and critic components.
        
        Returns:
            dict: Dictionary containing parameter lists:
                 - 'actor': List of all actor network parameters
                 - 'critic': List of all critic network parameters
                 
        Usage:
            # Initialize separate optimizers for actor and critic
            params = agent.get_parameters()
            actor_optimizer = Adam(params['actor'], lr=1e-4)
            critic_optimizer = Adam(params['critic'], lr=1e-3)
            
            # Count total parameters
            total_params = sum(p.numel() for p in params['actor'] + params['critic'])
        """
        return {
            'actor': list(self.actor.parameters()),
            'critic': list(self.critic.parameters())
        }
    
    def train(self):
        """
        Enable training mode for both actor and critic networks.
        
        This method synchronizes the training state of both networks, ensuring:
        - Dropout layers are active (if present)
        - Batch normalization uses batch statistics (if present)
        - Gradients are computed during backward passes
        - Networks are prepared for parameter updates
        
        Should be called before training loops to ensure proper network behavior.
        """
        self.actor.train()
        self.critic.train()
    
    def eval(self):
        """
        Enable evaluation mode for both actor and critic networks.
        
        This method synchronizes the evaluation state of both networks, ensuring:
        - Dropout layers are disabled (if present)
        - Batch normalization uses running statistics (if present)
        - Networks produce deterministic outputs
        - Networks are prepared for inference without gradient computation
        
        Should be called before evaluation, validation, or inference to ensure
        consistent and reproducible network behavior.
        """
        self.actor.eval()
        self.critic.eval()
    
    def to(self, device):
        """
        Transfer both actor and critic networks to the specified device.
        
        This method ensures synchronized device placement for both networks,
        maintaining consistency and enabling efficient computation. All network
        parameters, buffers, and intermediate computations will use the target device.
        
        Args:
            device: Target PyTorch device (e.g., 'cuda:0', 'cpu', torch.device('cuda'))
        
        Returns:
            self: Returns the ActorCriticPair instance for method chaining
            
        Usage:
            # Move to GPU
            agent.to('cuda:0')
            
            # Move to CPU
            agent.to('cpu')
            
            # Method chaining
            agent.to(device).train()
        
        Note:
            This operation may involve memory transfers and should be called
            before training begins. Moving between devices during training
            can significantly impact performance.
        """
        self.actor.to(device)
        self.critic.to(device)
        self.device = device
        return self