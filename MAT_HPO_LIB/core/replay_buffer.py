"""
Experience Replay Buffer for MAT-HPO

Sophisticated experience replay system designed for multi-agent hyperparameter 
optimization. This module provides efficient storage, sampling, and management 
of experience transitions to enable stable and effective reinforcement learning.

Key Features:
- Prioritized experience replay based on reward values
- Support for both single-agent and multi-agent scenarios
- Efficient memory management with automatic capacity control
- Comprehensive statistics tracking for monitoring and debugging
- Robust error handling for various reward formats

The replay buffer is crucial for breaking temporal correlations in experience
sequences and enabling stable learning in the SQDDPG algorithm.

Adapted and enhanced from the original MAT-HPO implementation with improved
robustness, documentation, and additional utility functions.
"""

import numpy as np
import random
from collections import namedtuple
from typing import List, Any, Optional


# Define transition structure
Transition = namedtuple('Transition', ['state', 'action', 'reward'])


class TransReplayBuffer:
    """
    Advanced Transition Replay Buffer with prioritized sampling capabilities.
    
    This class implements an experience replay buffer specifically designed for
    hyperparameter optimization tasks. It stores transitions (state, action, reward)
    and provides intelligent sampling strategies that improve learning efficiency.
    
    Key Features:
    - Prioritized sampling: Favors high-reward transitions for faster convergence
    - Automatic capacity management: Maintains fixed buffer size via FIFO eviction
    - Flexible reward handling: Supports scalar and vector rewards
    - Statistical monitoring: Tracks buffer utilization and reward distributions
    - Fallback mechanisms: Graceful handling of edge cases and errors
    
    The prioritized sampling strategy is particularly effective for HPO because:
    1. High-performing configurations provide more informative gradients
    2. Rare good configurations need to be retained longer in memory
    3. Exploration can be guided by previously successful parameter combinations
    """
    
    def __init__(self, size: int):
        """
        Initialize the replay buffer with specified capacity.
        
        Args:
            size: Maximum number of transitions to store. When this limit is reached,
                 oldest transitions are automatically removed (FIFO policy).
                 Typical values range from 1000-50000 depending on problem complexity.
                 
        Raises:
            ValueError: If size <= 0
        """
        if size <= 0:
            raise ValueError("Buffer size must be positive")
        self.size = size
        self.buffer = []
        self.Transition = Transition
        
    def add_experience(self, transition: Transition):
        """
        Add a new experience transition to the buffer with automatic capacity management.
        
        This method implements a First-In-First-Out (FIFO) eviction policy when
        the buffer reaches maximum capacity. This ensures:
        1. Recent experiences are always available for learning
        2. Memory usage remains bounded and predictable
        3. Very old experiences (which may be less relevant) are naturally discarded
        
        Args:
            transition: Transition namedtuple containing:
                       - state: The state representation when action was taken
                       - action: The action(s) executed by the agent(s)
                       - reward: The reward received (scalar or vector)
                       
        Note:
            The transition should match the expected format defined by the
            Transition namedtuple to ensure compatibility with sampling methods.
        """
        # Remove oldest transition if buffer is full
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
            
        self.buffer.append(transition)
    
    def get_single(self, index: int) -> Transition:
        """
        Get a single transition by index.
        
        Args:
            index: Index of the transition to retrieve
            
        Returns:
            Transition at the specified index
        """
        return self.buffer[index]
    
    def get_batch(self, batch_size: int) -> List[Transition]:
        """
        Sample a batch using intelligent prioritized sampling strategy.
        
        This method implements a sophisticated sampling approach that balances
        exploitation of high-reward experiences with exploration diversity:
        
        Algorithm:
        1. Sort all transitions by reward value (highest first)
        2. Select top 150% of requested batch size to create priority pool
        3. Randomly sample from this priority pool to maintain diversity
        4. Fallback to uniform sampling if reward-based sorting fails
        
        Benefits:
        - Prioritizes informative high-reward transitions
        - Maintains exploration through randomization within top transitions
        - Prevents overfitting to a single high-reward configuration
        - Robust handling of various reward formats (scalar, vector, complex)
        
        Args:
            batch_size: Number of transitions to sample. If buffer contains fewer
                       transitions, returns all available transitions.
            
        Returns:
            List of sampled transitions, prioritizing high-reward experiences
            while maintaining diversity through controlled randomization.
            
        Note:
            If reward extraction fails (e.g., incompatible reward format),
            automatically falls back to uniform random sampling.
        """
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        
        # Sort by reward (highest first) - assumes reward is a single value or array
        try:
            sorted_buffer = sorted(
                self.buffer, 
                key=lambda x: x.reward[0] if hasattr(x.reward, '__getitem__') else x.reward,
                reverse=True
            )
        except (TypeError, IndexError):
            # Fallback to regular sampling if sorting fails
            return random.sample(self.buffer, batch_size)
        
        # Take top portion for prioritized sampling
        top_size = min(int(batch_size * 1.5), len(sorted_buffer))
        top_buffer = sorted_buffer[:top_size]
        
        # Random sample from top transitions
        if len(top_buffer) <= batch_size:
            return top_buffer
        else:
            indices = np.random.choice(len(top_buffer), batch_size, replace=False)
            return [top_buffer[i] for i in indices]
    
    def get_random_batch(self, batch_size: int) -> List[Transition]:
        """
        Sample a batch of transitions using uniform random sampling.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of randomly sampled transitions
        """
        if len(self.buffer) <= batch_size:
            return self.buffer.copy()
        
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        """Clear all transitions from the buffer."""
        self.buffer = []
    
    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if the buffer is at maximum capacity."""
        return len(self.buffer) >= self.size
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the replay buffer contents.
        
        Returns:
            Dictionary containing buffer statistics
        """
        if not self.buffer:
            return {
                'size': 0,
                'capacity': self.size,
                'utilization': 0.0,
                'avg_reward': 0.0,
                'max_reward': 0.0,
                'min_reward': 0.0
            }
        
        # Extract rewards
        rewards = []
        for transition in self.buffer:
            if hasattr(transition.reward, '__getitem__'):
                rewards.append(transition.reward[0])
            else:
                rewards.append(transition.reward)
        
        return {
            'size': len(self.buffer),
            'capacity': self.size,
            'utilization': len(self.buffer) / self.size,
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards)
        }


class MultiAgentReplayBuffer:
    """
    Advanced Multi-Agent Experience Management System.
    
    This class manages separate replay buffers for multiple agents, enabling
    specialized experience storage and sampling strategies per agent. This is
    particularly valuable in multi-agent hyperparameter optimization where:
    
    1. Different agents may have different learning rates and need different
       experience retention strategies
    2. Agent specialization requires isolated experience pools to prevent
       interference between different hyperparameter subspaces
    3. Individual agent performance can be monitored and debugged separately
    4. Heterogeneous agent architectures may benefit from different buffer sizes
    
    Features:
    - Independent buffer management per agent
    - Centralized control and monitoring interface
    - Agent-specific statistics and performance tracking
    - Efficient memory allocation and cleanup
    - Robust error handling with agent ID validation
    
    Use Cases:
    - Multi-agent hyperparameter optimization (like SQDDPG)
    - Heterogeneous agent architectures
    - Specialized learning strategies per agent
    - Independent agent evaluation and debugging
    """
    
    def __init__(self, num_agents: int, buffer_size: int):
        """
        Initialize independent replay buffers for multiple agents.
        
        Creates isolated experience storage for each agent, allowing for
        specialized learning and experience management strategies.
        
        Args:
            num_agents: Number of agents requiring separate buffers.
                       Must be positive integer. Typical values: 2-10.
            buffer_size: Capacity of each individual agent's buffer.
                        All agents get the same buffer size, but this can
                        be customized by directly accessing self.buffers.
                        
        Raises:
            ValueError: If num_agents <= 0 or buffer_size <= 0
        """
        if num_agents <= 0:
            raise ValueError("Number of agents must be positive")
        if buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
        self.num_agents = num_agents
        self.buffers = [TransReplayBuffer(buffer_size) for _ in range(num_agents)]
    
    def add_experience(self, agent_id: int, transition: Transition):
        """Add experience to a specific agent's buffer."""
        if 0 <= agent_id < self.num_agents:
            self.buffers[agent_id].add_experience(transition)
        else:
            raise ValueError(f"Invalid agent_id: {agent_id}. Must be 0-{self.num_agents-1}")
    
    def get_batch(self, agent_id: int, batch_size: int) -> List[Transition]:
        """Get batch from a specific agent's buffer."""
        if 0 <= agent_id < self.num_agents:
            return self.buffers[agent_id].get_batch(batch_size)
        else:
            raise ValueError(f"Invalid agent_id: {agent_id}. Must be 0-{self.num_agents-1}")
    
    def clear_all(self):
        """Clear all agent buffers."""
        for buffer in self.buffers:
            buffer.clear()
    
    def get_all_statistics(self) -> dict:
        """Get statistics for all agent buffers."""
        stats = {}
        for i, buffer in enumerate(self.buffers):
            stats[f'agent_{i}'] = buffer.get_statistics()
        return stats