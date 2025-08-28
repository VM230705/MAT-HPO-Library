"""
SQDDPG: Shapley Q-Value Deep Deterministic Policy Gradient

A sophisticated multi-agent reinforcement learning algorithm designed specifically 
for hyperparameter optimization. This implementation uses three agents where each 
agent controls different subsets of hyperparameters, and Shapley values are used 
to fairly distribute credit among agents for their contributions to the overall 
performance.

The algorithm combines:
- Deep Deterministic Policy Gradient (DDPG) for continuous action spaces
- Shapley value-based credit assignment for multi-agent cooperation
- Transformer-based actor-critic architecture for complex state-action relationships

Adapted and enhanced from the original MAT-HPO implementation with improved
stability, documentation, and flexibility.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from torch.distributions import Normal
from typing import Optional, Tuple, List
import math

from .agent import Actor, Critic, get_device


class SQDDPG:
    """
    Shapley Q-Value Deep Deterministic Policy Gradient for multi-agent optimization.
    
    Uses Shapley values to distribute credit among multiple agents, each responsible
    for different subsets of hyperparameters.
    """
    
    def __init__(self, 
                 hyp_num0: int,
                 hyp_num1: int, 
                 hyp_num2: int,
                 sample_size: int = 10,
                 device: Optional[torch.device] = None):
        """
        Initialize SQDDPG with three agents.
        
        Args:
            hyp_num0: Number of hyperparameters for agent 0
            hyp_num1: Number of hyperparameters for agent 1
            hyp_num2: Number of hyperparameters for agent 2
            sample_size: Number of coalition samples for Shapley value computation
            device: Device to place models on
        """
        # Input validation
        if any(h <= 0 for h in [hyp_num0, hyp_num1, hyp_num2]):
            raise ValueError("All hyperparameter counts must be positive")
        if sample_size < 1:
            raise ValueError("Sample size must be at least 1")
            
        # Device setup with automatic GPU detection
        self.device = device if device is not None else get_device()
        self.sample_size = sample_size
        
        # Named tuple for structured experience transitions
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward'))
        
        # Agent configuration - fixed at 3 agents for this implementation
        self.n_ = 3  # Number of cooperative agents
        self.hyp_num0, self.hyp_num1, self.hyp_num2 = hyp_num0, hyp_num1, hyp_num2
        
        # Standardize dimensions across agents for consistent tensor operations
        # All agents work with the same action/observation space size
        self.act_dim = max(hyp_num0, hyp_num1, hyp_num2)  # Maximum dimension for padding
        self.obs_dim = self.act_dim  # Observation space matches action space
        
        # Create actors (one per agent)
        self.actor0 = Actor(self.hyp_num0, device=self.device)
        self.actor1 = Actor(self.hyp_num1, device=self.device) 
        self.actor2 = Actor(self.hyp_num2, device=self.device)
        
        # Create critics (one per agent, input = concatenated obs+actions of all agents)
        critic_input_dim = (self.act_dim + self.obs_dim) * self.n_
        self.critic0 = Critic(critic_input_dim, device=self.device)
        self.critic1 = Critic(critic_input_dim, device=self.device)
        self.critic2 = Critic(critic_input_dim, device=self.device)
        
        # Create masks for different agents (to handle different hyperparameter counts)
        self.masks = self._create_masks()
        
    def _create_masks(self) -> List[torch.Tensor]:
        """Create masks to handle different hyperparameter dimensions per agent"""
        masks = []
        
        # Agent 0 mask
        mask0 = torch.zeros(self.act_dim, dtype=torch.float32, device=self.device)
        mask0[:self.hyp_num0] = 1.0
        masks.append(mask0)
        
        # Agent 1 mask
        mask1 = torch.zeros(self.act_dim, dtype=torch.float32, device=self.device)
        mask1[:self.hyp_num1] = 1.0
        masks.append(mask1)
        
        # Agent 2 mask
        mask2 = torch.zeros(self.act_dim, dtype=torch.float32, device=self.device)
        mask2[:self.hyp_num2] = 1.0
        masks.append(mask2)
        
        return masks
    
    def policy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Generate actions for all agents given the current state.
        
        Args:
            state: State tensor [batch_size, n_agents, obs_dim]
            
        Returns:
            actions: Action tensor [batch_size, n_agents, act_dim]
        """
        batch_size = state.size(0)
        
        # Handle state dimension - if 2D, expand to 3D
        if len(state.shape) == 2:
            # Assume state is [batch_size, obs_dim], replicate for each agent
            state = state.unsqueeze(1).expand(batch_size, self.n_, -1)
        
        # Generate actions for each agent
        action0 = (self.actor0(state[:, 0, :self.hyp_num0]) * self.masks[0][:self.hyp_num0]).unsqueeze(1)
        action1 = (self.actor1(state[:, 1, :self.hyp_num1]) * self.masks[1][:self.hyp_num1]).unsqueeze(1)
        action2 = (self.actor2(state[:, 2, :self.hyp_num2]) * self.masks[2][:self.hyp_num2]).unsqueeze(1)
        
        # Pad actions to same dimension
        if self.hyp_num0 < self.act_dim:
            padding = torch.zeros(batch_size, 1, self.act_dim - self.hyp_num0, device=self.device)
            action0 = torch.cat([action0, padding], dim=2)
            
        if self.hyp_num1 < self.act_dim:
            padding = torch.zeros(batch_size, 1, self.act_dim - self.hyp_num1, device=self.device)
            action1 = torch.cat([action1, padding], dim=2)
            
        if self.hyp_num2 < self.act_dim:
            padding = torch.zeros(batch_size, 1, self.act_dim - self.hyp_num2, device=self.device)
            action2 = torch.cat([action2, padding], dim=2)
        
        # Concatenate all agent actions
        actions = torch.cat([action0, action1, action2], dim=1)
        
        return actions
    
    def select_action(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Select actions from policy logits (deterministic for now).
        
        Args:
            logits: Action logits from policy
            
        Returns:
            Selected actions
        """
        return logits.to(self.device)
    
    def unpack_data(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unpack batch data for training.
        
        Args:
            batch: Batch of transitions
            
        Returns:
            Tuple of (rewards, actions, states)
        """
        batch_size = len(batch.state)
        
        rewards = torch.tensor(batch.reward, dtype=torch.float, device=self.device)
        actions = torch.tensor(np.stack([t for t in batch.action], axis=0), 
                              dtype=torch.float, device=self.device)
        states = torch.tensor(np.stack([t for t in batch.state], axis=0),
                             dtype=torch.float, device=self.device)
        
        return rewards, actions, states
    
    def sample_grand_coalitions(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample grand coalitions for Shapley value computation.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Tuple of (subcoalition_map, grand_coalitions)
        """
        # Create sequential coalition mask
        seq_set = torch.tril(torch.ones(self.n_, self.n_), diagonal=0).to(self.device)
        
        # Sample random permutations of agents (grand coalitions)
        grand_coalitions_pos = torch.multinomial(
            torch.ones(batch_size * self.sample_size, self.n_) / self.n_,
            self.n_, 
            replacement=False
        ).to(self.device)
        
        # Create individual agent map
        individual_map = torch.zeros(batch_size * self.sample_size * self.n_, self.n_).to(self.device)
        individual_map.scatter_(1, grand_coalitions_pos.contiguous().view(-1, 1), 1)
        individual_map = individual_map.contiguous().view(
            batch_size, self.sample_size, self.n_, self.n_)
        
        # Create subcoalition map
        subcoalition_map = torch.matmul(individual_map, seq_set)
        
        # Fix grand coalition construction
        offset = (torch.arange(batch_size * self.sample_size) * self.n_).reshape(-1, 1).to(self.device)
        grand_coalitions_pos_alter = grand_coalitions_pos + offset
        grand_coalitions = torch.zeros_like(grand_coalitions_pos_alter.flatten()).to(self.device)
        grand_coalitions[grand_coalitions_pos_alter.flatten()] = torch.arange(
            batch_size * self.sample_size * self.n_).to(self.device)
        grand_coalitions = grand_coalitions.reshape(batch_size * self.sample_size, self.n_) - offset
        
        grand_coalitions = grand_coalitions.unsqueeze(1).expand(
            batch_size * self.sample_size, self.n_, self.n_
        ).contiguous().view(batch_size, self.sample_size, self.n_, self.n_)
        
        return subcoalition_map, grand_coalitions
    
    def marginal_contribution(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        Compute marginal contributions using Shapley values.
        
        Args:
            obs: Observations tensor
            act: Actions tensor
            
        Returns:
            Shapley values for each agent
        """
        batch_size = obs.size(0)
        subcoalition_map, grand_coalitions = self.sample_grand_coalitions(batch_size)
        
        # Expand grand coalitions for actions
        grand_coalitions = grand_coalitions.unsqueeze(-1).expand(
            batch_size, self.sample_size, self.n_, self.n_, self.act_dim)
        
        # Reorder actions according to grand coalitions
        act = act.unsqueeze(1).unsqueeze(2).expand(
            batch_size, self.sample_size, self.n_, self.n_, self.act_dim
        ).gather(3, grand_coalitions)
        
        # Apply coalition masks
        act_map = subcoalition_map.unsqueeze(-1).float()
        act = act * act_map
        act = act.contiguous().view(batch_size, self.sample_size, self.n_, -1)
        
        # Prepare observations
        obs = obs.unsqueeze(1).unsqueeze(2).expand(
            batch_size, self.sample_size, self.n_, self.n_, self.obs_dim)
        obs = obs.contiguous().view(batch_size, self.sample_size, self.n_, self.n_ * self.obs_dim)
        
        # Combine observations and actions
        inp = torch.cat((obs, act), dim=-1)
        
        # Compute values using each critic
        values = [
            self.critic0(inp[:, :, 0, :]),
            self.critic1(inp[:, :, 1, :]), 
            self.critic2(inp[:, :, 2, :])
        ]
        values = torch.stack(values, dim=2)
        
        return values
    
    def get_loss(self, batch) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Compute losses for actors and critics.
        
        Args:
            batch: Batch of transitions
            
        Returns:
            Tuple of (action_losses, value_losses, log_probs)
        """
        batch_size = len(batch.state)
        n = self.n_
        
        rewards, actions, states = self.unpack_data(batch)
        
        # Generate actions for current policy
        action_out = self.policy(states)
        actions_current = self.select_action(action_out)
        
        # Compute Shapley values for current policy actions
        shapley_values = self.marginal_contribution(states, actions_current).mean(dim=1)
        shapley_values = shapley_values.contiguous().view(-1, n)
        
        # Compute Shapley values for exploration actions
        shapley_values_sum = self.marginal_contribution(states, actions).mean(dim=1)
        shapley_values_sum = shapley_values_sum.contiguous().view(-1, n).sum(dim=-1, keepdim=True)
        shapley_values_sum = shapley_values_sum.expand(batch_size, self.n_)
        
        # Compute advantages and losses
        deltas = rewards - shapley_values_sum
        advantages = shapley_values
        
        # Actor losses (negative advantages for gradient ascent)
        action_losses = [-advantages[:, i].mean() for i in range(n)]
        
        # Critic losses (squared deltas)
        value_losses = [deltas[:, i].pow(2).mean() for i in range(n)]
        
        return action_losses, value_losses, action_out