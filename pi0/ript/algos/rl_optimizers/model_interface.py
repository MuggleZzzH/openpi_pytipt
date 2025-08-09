from abc import ABC, abstractmethod
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch.nn.functional as F
class RLModelInterface(ABC):
    """
    Abstract interface for models to work with the RL optimizer.
    
    This interface provides a standard way to interact with different types of RL models,
    focusing on the core functionality needed for reinforcement learning:
    - Computing action log probabilities
    """
    def __init__(self, model, invalid_logprob=1.0, **kwargs):
        self.model = model
        self.invalid_logprob = invalid_logprob

    @abstractmethod
    def compute_act_logits(self, model, episodes: List[Dict[str, Any]], device: Optional[torch.device] = None):
        """
        Compute log probabilities of actions.

        Args:
            model: Model to compute logits with (could be policy model)
            episodes: List of episode dictionaries
            device: Device to place tensors on (optional)
            
        Returns:
            Log probabilities of the actions
        """
        pass
    
    @property
    @abstractmethod
    def device(self):
        """Return the device the model is on"""
        pass
    
    @abstractmethod
    def get_policy_model(self):
        """Return the policy model"""
        pass
    
    def process_episodes(self, episodes: List[Dict[str, Any]], device: Optional[torch.device] = None):
        """
        Process episodes to extract and pad context tokens, action indices, and create action token mask.
        
        Args:
            episodes: List of episode dictionaries
            device: Device to place tensors on (optional)
        
        Returns:
            Tuple containing information for action logits computation
        """
        pass


class QuestModelAdapter(RLModelInterface):
    """
    Adapter for Quest models that use a token-based approach.
    This works with models like QueST_rl that use context tokens and action indices.
    """
    def __init__(self, model, invalid_logprob=1.0, **kwargs):
        """
        Initialize the adapter with a Quest model.
        
        Args:
            model: Quest model
            invalid_logprob: Value to use for invalid log probabilities
        """
        super().__init__(model, invalid_logprob, **kwargs)
        self.valid_action_steps = kwargs.get('valid_action_steps', None)

    def process_episodes(self, episodes: List[Dict[str, Any]], device: Optional[torch.device] = None, max_seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Process episodes to extract and pad context tokens, action indices, and create action token mask.
        
        Args:
            episodes: List of episode dictionaries
            device: Device to place tensors on (optional)
            
        Returns:
            Tuple containing:
                - Padded context tokens tensor
                - Padded action indices tensor
                - Action token mask tensor
        """
        if not episodes:
            # Return empty tensors if no episodes
            if device is None:
                device = torch.device("cpu")
            return (
                torch.empty((0, 0, 0), device=device),
                torch.empty((0, 0), device=device),
                torch.empty((0, 0, 0), dtype=torch.bool, device=device)
            )
        
        # Extract context tokens and action indices
        all_context_tokens = [torch.tensor(episode['context_tokens']) for episode in episodes]
        all_action_indices = [torch.tensor(episode['action_indices']) for episode in episodes]
        
        # Pad context tokens
        max_seq_len = max_seq_len if max_seq_len is not None else max(t.size(0) for t in all_context_tokens)
        padded_all_context_tokens = []
        for tensor in all_context_tokens:
            pad_length = max_seq_len - tensor.size(0)
            if pad_length > 0:
                tensor = F.pad(tensor, (0, 0, 0, 0, 0, pad_length))
            padded_all_context_tokens.append(tensor)
        all_context_tokens = torch.stack(padded_all_context_tokens, dim=0)
        
        # Pad action indices
        padded_all_action_indices = []
        for tensor in all_action_indices:
            pad_length = max_seq_len - tensor.size(0)
            if pad_length > 0:
                tensor = F.pad(tensor, (0, 0, 0, pad_length))
            padded_all_action_indices.append(tensor)
        all_action_indices = torch.stack(padded_all_action_indices, dim=0).long()
        
        # Create action token mask
        B, T, A = all_action_indices.shape
        all_action_token_mask = torch.zeros(B, T, A, dtype=torch.bool)
        for i, episode in enumerate(episodes):
            if self.valid_action_steps is None:
                valid_steps = all_action_token_mask.shape[2] # all steps are valid
            else:
                valid_steps = self.valid_action_steps
            all_action_token_mask[i, :episode['policy_inference_steps'][0], :valid_steps] = True
        
        # Move tensors to device if specified
        if device is not None:
            all_context_tokens = all_context_tokens.to(device)
            all_action_indices = all_action_indices.to(device)
            all_action_token_mask = all_action_token_mask.to(device)
        
        return all_context_tokens, all_action_indices, all_action_token_mask, max_seq_len

    def compute_act_logits(self, model, episodes, device: Optional[torch.device] = None, max_seq_len: Optional[int] = None):
        """
        Compute action log probabilities using a flattened approach, following QueST_rl implementation.
        
        Args:
            model: Model to compute logits with
            episodes: Either a list of episode dictionaries or a tuple of (context_tokens, action_indices, action_token_mask)
            device: Device to place tensors on (optional)
            max_seq_len: Maximum sequence length to use for logits computation; if None, inference from episodes
            
        Returns:
            Log probabilities (B, T*A)
        """

        context_tokens, action_indices, action_token_mask, max_seq_len = self.process_episodes(episodes, device, max_seq_len)

        device = self.device if device is None else device
        B, T, A = action_indices.shape
        flat_bz = B * T
        context_len = context_tokens.shape[-2]
        flat_context_tokens = context_tokens.view(flat_bz, context_len, -1)
        flat_action_indices = action_indices.view(flat_bz, -1)
        flat_start_token = torch.ones((flat_bz, 1), device=device, dtype=torch.long) * model.start_token
        flat_step_mask = action_token_mask.any(dim=-1).view(flat_bz) # (B * T)

        flat_logprobs = torch.ones(flat_bz, A, device=device) * self.invalid_logprob

        flat_context_tokens_valid = flat_context_tokens[flat_step_mask]
        flat_action_indices_valid = flat_action_indices[flat_step_mask]
        flat_start_token_valid = flat_start_token[flat_step_mask]
        flat_input_token_valid = torch.cat([flat_start_token_valid, flat_action_indices_valid], dim=1)
        flat_logits_valid = model(flat_input_token_valid, flat_context_tokens_valid)[:, :-1, :]
        flat_logprobs_valid = torch.nn.functional.log_softmax(flat_logits_valid, dim=-1)
        flat_logprobs_valid = torch.gather(flat_logprobs_valid, dim=-1, index=flat_action_indices_valid[..., None])[..., 0]
      
        flat_logprobs[flat_step_mask] = flat_logprobs_valid

        logprobs = flat_logprobs.view(B, T, A)
        sequence_logprobs = logprobs.view(B, T * A)
        return sequence_logprobs, max_seq_len
    
    @property
    def device(self):
        return self.model.device
    
    def get_policy_model(self):
        return self.model.policy_prior