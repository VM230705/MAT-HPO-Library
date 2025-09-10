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
            state: State tensor - can be [batch_size, features] or [batch_size, n_agents, obs_dim]
            
        Returns:
            actions: Action tensor [batch_size, n_agents, act_dim]
        """
        batch_size = state.size(0)
        
        # Handle different state formats
        if len(state.shape) == 2:
            # Format: [batch_size, features] -> need to convert to [batch_size, n_agents, obs_dim]
            state_features = state.size(1)  # e.g., 12 features
            
            # Split state features across agents or replicate for each agent
            if state_features >= self.obs_dim * self.n_:
                # If we have enough features, split them across agents
                feature_per_agent = state_features // self.n_
                agent_states = []
                for i in range(self.n_):
                    start_idx = i * feature_per_agent
                    end_idx = start_idx + min(feature_per_agent, self.obs_dim)
                    agent_state = state[:, start_idx:end_idx]
                    # Pad if necessary
                    if agent_state.size(1) < self.obs_dim:
                        padding = torch.zeros(batch_size, self.obs_dim - agent_state.size(1), device=self.device)
                        agent_state = torch.cat([agent_state, padding], dim=1)
                    agent_states.append(agent_state.unsqueeze(1))
                state = torch.cat(agent_states, dim=1)
            else:
                # If not enough features, replicate the state for each agent
                # First pad state to obs_dim if necessary
                if state_features < self.obs_dim:
                    padding = torch.zeros(batch_size, self.obs_dim - state_features, device=self.device)
                    state = torch.cat([state, padding], dim=1)
                elif state_features > self.obs_dim:
                    # Truncate if too many features
                    state = state[:, :self.obs_dim]
                # Now replicate for each agent
                state = state.unsqueeze(1).expand(batch_size, self.n_, -1)
                
        elif len(state.shape) == 3:
            # Format: [batch_size, current_agents, features] -> need to convert to [batch_size, n_agents, obs_dim]
            current_agents = state.size(1)
            current_features = state.size(2)
            
            if current_agents == 1:
                # Single agent state - replicate across all agents
                single_agent_state = state.squeeze(1)  # [batch_size, features]
                
                # Pad/truncate to obs_dim
                if current_features < self.obs_dim:
                    padding = torch.zeros(batch_size, self.obs_dim - current_features, device=self.device)
                    single_agent_state = torch.cat([single_agent_state, padding], dim=1)
                elif current_features > self.obs_dim:
                    single_agent_state = single_agent_state[:, :self.obs_dim]
                    
                # Replicate for all agents
                state = single_agent_state.unsqueeze(1).expand(batch_size, self.n_, -1)
                
            elif current_agents == self.n_:
                # Already correct number of agents - just adjust feature dimension
                if current_features != self.obs_dim:
                    agent_states = []
                    for i in range(self.n_):
                        agent_state = state[:, i, :]  # [batch_size, features]
                        
                        # Pad/truncate to obs_dim
                        if current_features < self.obs_dim:
                            padding = torch.zeros(batch_size, self.obs_dim - current_features, device=self.device)
                            agent_state = torch.cat([agent_state, padding], dim=1)
                        elif current_features > self.obs_dim:
                            agent_state = agent_state[:, :self.obs_dim]
                            
                        agent_states.append(agent_state.unsqueeze(1))
                    state = torch.cat(agent_states, dim=1)
            else:
                # Incorrect number of agents - flatten and redistribute
                flattened_state = state.view(batch_size, -1)  # [batch_size, current_agents * current_features]
                total_features = flattened_state.size(1)
                
                if total_features >= self.obs_dim * self.n_:
                    # Split across agents
                    feature_per_agent = total_features // self.n_
                    agent_states = []
                    for i in range(self.n_):
                        start_idx = i * feature_per_agent
                        end_idx = start_idx + min(feature_per_agent, self.obs_dim)
                        agent_state = flattened_state[:, start_idx:end_idx]
                        
                        if agent_state.size(1) < self.obs_dim:
                            padding = torch.zeros(batch_size, self.obs_dim - agent_state.size(1), device=self.device)
                            agent_state = torch.cat([agent_state, padding], dim=1)
                        agent_states.append(agent_state.unsqueeze(1))
                    state = torch.cat(agent_states, dim=1)
                else:
                    # Replicate across agents
                    if total_features < self.obs_dim:
                        padding = torch.zeros(batch_size, self.obs_dim - total_features, device=self.device)
                        flattened_state = torch.cat([flattened_state, padding], dim=1)
                    elif total_features > self.obs_dim:
                        flattened_state = flattened_state[:, :self.obs_dim]
                    state = flattened_state.unsqueeze(1).expand(batch_size, self.n_, -1)
        
        # Verify final dimensions
        if len(state.shape) != 3 or state.size(1) != self.n_:
            raise ValueError(f"State dimension error: expected [batch_size, {self.n_}, {self.obs_dim}], got {state.shape}")
        
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
        
        # Handle observation dimensions - ensure obs matches expected format
        if len(obs.shape) == 3 and obs.size(1) == self.n_:
            # obs is [batch_size, n_agents, features] - extract the actual feature dimension
            actual_obs_dim = obs.size(2)
            if actual_obs_dim != self.obs_dim:
                # Adjust observations to match obs_dim
                if actual_obs_dim < self.obs_dim:
                    # Pad observations
                    padding = torch.zeros(batch_size, self.n_, self.obs_dim - actual_obs_dim, device=self.device)
                    obs = torch.cat([obs, padding], dim=2)
                else:
                    # Truncate observations
                    obs = obs[:, :, :self.obs_dim]
        elif len(obs.shape) == 2:
            # obs is [batch_size, features] - convert to multi-agent format manually
            obs_features = obs.size(1)
            
            # Pad/truncate to obs_dim
            if obs_features < self.obs_dim:
                padding = torch.zeros(batch_size, self.obs_dim - obs_features, device=self.device)
                obs = torch.cat([obs, padding], dim=1)
            elif obs_features > self.obs_dim:
                obs = obs[:, :self.obs_dim]
                
            # Replicate for all agents
            obs = obs.unsqueeze(1).expand(batch_size, self.n_, -1)
        elif len(obs.shape) == 3 and obs.size(1) == 1:
            # obs is [batch_size, 1, features] - replicate across agents
            obs_features = obs.size(2)
            obs = obs.squeeze(1)  # [batch_size, features]
            
            # Pad/truncate to obs_dim
            if obs_features < self.obs_dim:
                padding = torch.zeros(batch_size, self.obs_dim - obs_features, device=self.device)
                obs = torch.cat([obs, padding], dim=1)
            elif obs_features > self.obs_dim:
                obs = obs[:, :self.obs_dim]
                
            # Replicate for all agents
            obs = obs.unsqueeze(1).expand(batch_size, self.n_, -1)
        
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
        
        # Prepare observations - now obs should be [batch_size, n_agents, obs_dim]
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