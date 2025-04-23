#!/usr/bin/python3
"""LSTM Cells module

This module contains the implementation of an LSTM cell with an attention mechanism 
and other LSTM cells with the application of physical laws for the attention function.

Classes:
    LSTMCellWithAttention: A single LSTM cell that incorporates an attention mechanism
                           over the previous hidden states in a sequence.
    LSTMWithAttentionEntropyReg: An LSTM cell with an attention mechanism and an
                                 entropic regularization term applied to the attention weights.
    LSTMWithAttentionTimeDecay: An LSTM cell with an attention mechanism and a time decay
                           factor applied to the attention weights.
    LSTMWithAttentionMixed: An LSTM cell with a mixed attention mechanism that combines
                           time decay, entropy regularization, and randomness in the attention weights.
    LSTMWithAttentionPD: An LSTM cell with an attention mechanism that applies the potential difference principle.
    LSTMWithGravitationalAttention: An LSTM cell with an attention mechanism that applies gravitational principles.
    LSTMMomentumAttention: An LSTM cell with an attention mechanism that applies momentum principles.
    
    AttentionByPotentialDifferenceModule: A module that calculates attention weights based on the potential difference principle.
    LSTMCellWithAttentionPD: A single LSTM cell that incorporates an attention mechanism based on the potential difference principle.
    GravitationalAttention: A single LSTM cell that incorporates an attention mechanism based on gravitational principles.
    MomentumAttention: A single LSTM cell that incorporates an attention mechanism based on momentum principles.
    
Functions:
    lstm_cell_with_attention(input_tensor, hidden_state, ...): Calculation of a single step of an LSTM cell with an attention mechanism.
    lstm_cell_attention_entropy_reg(input_tensor, hidden_state, ...): Calculation of a single step of an LSTM cell with 
                                                                      an attention mechanism and entropic regularization.
    lstm_cell_attention_time_decay(input_tensor, hidden_state, ...): Calculation of a single step of an LSTM cell with an attention mechanism and time decay.
    lstm_cell_attention_mixed(input_tensor, hidden_state, ...): Calculation of a single step of an LSTM cell with an attention mechanism, 
                                                                entropic regularization, time decay, and a random element.
    attention_by_potential_difference(h_current, previous_hidden_states, ...): Calculation of attention weights based on the principle of potential difference.
    gravitational_attention_function(h_current, previous_hidden_states, ...): Calculation of attention weights based on an analogy with the law of gravity.
    momentum_attention_function(h_current, previous_hidden_states, ...): Calculation of attention weights based on the principle of momentum.
    
Examples:
>>> import torch
>>> import torch.nn as nn
>>> import torch.optim as optim
>>> import torch.nn.functional as F
>>> import numpy as np
>>> import random

>>> # Experimetal test LSTM settings
>>> seed = 3
>>> random.seed(seed)
>>> np.random.seed(seed)
>>> _ = torch.manual_seed(seed)

>>> ### Example usage LSTM with attention ###
>>> input_size = 10
>>> hidden_size = 20
>>> output_size = 5
>>> # Initialize the model
>>> lstm_cell_attention = LSTMCellWithAttention(input_size, hidden_size, output_size)

>>> # Dummy input: (batch_size, seq_length, input_size)
>>> batch_size = 5
>>> seq_length = 3
>>> dummy_input = torch.randn(batch_size, seq_length, input_size)

>>> # Optional: Initialize hidden and cell states (if not provided, they'll be zero-initialized)
>>> h_prev = torch.zeros(batch_size, hidden_size)
>>> c_prev = torch.zeros(batch_size, hidden_size)

>>> # Forward pass
>>> outputs, h_t, c_t = lstm_cell_attention(dummy_input, h_prev, c_prev)

>>> # Print the outputs
>>> print("Outputs:", outputs.shape)
Outputs: torch.Size([5, 5])
>>> print("Final Hidden State:", h_t.shape)
Final Hidden State: torch.Size([5, 20])
>>> print("Final Cell State:", c_t.shape)
Final Cell State: torch.Size([5, 20])

>>> ### Example usage LSTM with attention and entropic regularization ###
>>> input_size = 10
>>> hidden_size = 20
>>> output_size = 1
>>> lambda_entropy = 0.1  # Example value for entropic regularization coefficient

>>> lstm_attention_entropy = LSTMWithAttentionEntropyReg(input_size, hidden_size, output_size, lambda_entropy)

>>> # Dummy input data: (batch_size, seq_length, input_size)
>>> batch_size = 5
>>> seq_length = 3
>>> dummy_input = torch.randn(batch_size, seq_length, input_size)

>>> # Initialize hidden and cell states
>>> h_prev = torch.zeros(batch_size, hidden_size)
>>> c_prev = torch.zeros(batch_size, hidden_size)

>>> # Forward pass
>>> outputs, h_t, c_t = lstm_attention_entropy(dummy_input, h_prev, c_prev)

>>> # Print the outputs
>>> print("Outputs shape:", outputs.shape)
Outputs shape: torch.Size([5, 1])
>>> print("Final Hidden shape:", h_t.shape)
Final Hidden shape: torch.Size([5, 20])
>>> print("Final Cell shape:", c_t.shape)
Final Cell shape: torch.Size([5, 20])

>>> ### Example usage LSTM with attention and time decay ###
>>> input_size = 10
>>> hidden_size = 20
>>> output_size = 1
>>> time_decay = 0.1  # Example value for time decay coefficient

>>> lstm_attention_decay = LSTMWithAttentionTimeDecay(input_size, hidden_size, output_size, time_decay)

>>> # Dummy input data: (batch_size, seq_length, input_size)
>>> batch_size = 5
>>> seq_length = 3
>>> dummy_input = torch.randn(batch_size, seq_length, input_size)

>>> # Initialize hidden and cell states
>>> h_prev = torch.zeros(batch_size, hidden_size)
>>> c_prev = torch.zeros(batch_size, hidden_size)

>>> # Forward pass
>>> outputs, h_t, c_t = lstm_attention_decay(dummy_input, h_prev, c_prev)

>>> # Print the outputs
>>> print("Outputs shape:", outputs.shape)
Outputs shape: torch.Size([5, 1])
>>> print("Final Hidden shape:", h_t.shape)
Final Hidden shape: torch.Size([5, 20])
>>> print("Final Cell shape:", c_t.shape)
Final Cell shape: torch.Size([5, 20])

>>> ### Example usage LSTM with attention and time decay and entropic regularization and randomness ###
>>> input_size = 10
>>> hidden_size = 20
>>> output_size = 1
>>> lambda_entropy = 0.01  # Example value for entropic regularization coefficient
>>> time_decay = 0.05  # Example value for time decay coefficient
>>> random_prob = 0.1  # Example value for randomness probability

>>> lstm_attention_mixed_model = LSTMWithAttentionMixed(input_size, hidden_size, output_size, lambda_entropy, time_decay, random_prob)

>>> # Dummy input data: (batch_size, seq_length, input_size)
>>> batch_size = 5
>>> seq_length = 3
>>> dummy_input = torch.randn(batch_size, seq_length, input_size)

>>> # Initialize hidden and cell states
>>> h_prev = torch.zeros(batch_size, hidden_size)
>>> c_prev = torch.zeros(batch_size, hidden_size)

>>> # Forward pass
>>> outputs, h_t, c_t = lstm_attention_mixed_model(dummy_input, h_prev, c_prev)

>>> # Print the outputs
>>> print("Outputs shape:", outputs.shape)
Outputs shape: torch.Size([5, 1])
>>> print("Final Hidden shape:", h_t.shape)
Final Hidden shape: torch.Size([5, 20])
>>> print("Final Cell shape:", c_t.shape)
Final Cell shape: torch.Size([5, 20])

>>> ### Example usage LSTM with attention and potential difference principle ###
>>> input_size = 10
>>> hidden_size = 20
>>> output_size = 1
>>> batch_size = 5
>>> seq_length = 3
>>> potential_type = 'norm'  # Example value for potential type
>>> time_resistance_lambda = 0.1  # Example value for time resistance coefficient
>>> semantic_resistance_lambda = 1.0  # Example value for semantic resistance coefficient

>>> # Initialize model
>>> lstm_attention_pd = LSTMWithAttentionPD(input_size=input_size, hidden_size=hidden_size, output_size=output_size, potential_type=potential_type, time_resistance_lambda=time_resistance_lambda, semantic_resistance_lambda=semantic_resistance_lambda)

>>> # Dummy input data
>>> dummy_input = torch.randn(batch_size, seq_length, input_size)
>>> h_prev = torch.zeros(batch_size, hidden_size)
>>> c_prev = torch.zeros(batch_size, hidden_size)

>>> # Forward pass
>>> outputs, h_t, c_t, attention_history = lstm_attention_pd(dummy_input, h_prev, c_prev)

>>> # Print the outputs
>>> print("Outputs shape:", outputs.shape)
Outputs shape: torch.Size([5, 1])
>>> print("Final Hidden State:", h_t.shape)
Final Hidden State: torch.Size([5, 20])
>>> print("Final Cell State:", c_t.shape)
Final Cell State: torch.Size([5, 20])
>>> print("Attention History:", attention_history.shape)
Attention History: torch.Size([5, 3, 3])

>>> ### Example usage LSTM with gravitational attention ###
>>> input_size = 10
>>> hidden_size = 20
>>> output_size = 1
>>> batch_size = 5
>>> seq_length = 3
>>> gravity_constant = 0.5  # Example value for gravitational constant
>>> distance_power = 2.5  # Example value for distance power
>>> weight_type = 'norm'  # Example value for weight type

>>> # Initialize model
>>> lstm_gravitational_attention = LSTMWithGravitationalAttention(input_size=input_size, hidden_size=hidden_size, output_size=output_size, gravity_constant=gravity_constant, distance_power=distance_power, weight_type=weight_type)

>>> # Dummy input data
>>> dummy_input = torch.randn(batch_size, seq_length, input_size)
>>> h_prev = torch.zeros(batch_size, hidden_size)
>>> c_prev = torch.zeros(batch_size, hidden_size)

>>> # Forward pass
>>> outputs, h_t, c_t, attention_history = lstm_gravitational_attention(dummy_input, h_prev, c_prev)

>>> # Print the outputs
>>> print("Outputs shape:", outputs.shape)
Outputs shape: torch.Size([5, 1])
>>> print("Final Hidden State:", h_t.shape)
Final Hidden State: torch.Size([5, 20])
>>> print("Final Cell State:", c_t.shape)
Final Cell State: torch.Size([5, 20])
>>> print("Attention History:", attention_history.shape)
Attention History: torch.Size([5, 3, 3])

>>> ### Example usage LSTM with momentum attention ###
>>> hidden_size = 20
>>> batch_size = 5
>>> seq_len = 3

>>> # Dummy hidden states
>>> h_current = torch.randn(batch_size, hidden_size)
>>> h_prev = torch.randn(batch_size, hidden_size)
>>> previous_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
>>> weight_type = 'norm'  # Example value for weight type
>>> distance_type = 'time'  # Example value for distance type
>>> distance_lambda = 0.05  # Example value for distance lambda

>>> # Initialization of the momentum attention module
>>> momentum_attn = LSTMMomentumAttention(hidden_size=hidden_size, weight_type=weight_type, distance_type=distance_type, distance_lambda=distance_lambda)

>>> # Performing a forward pass
>>> context_vector, attention_weights = momentum_attn(h_current, previous_hidden_states, h_prev)

>>> # Print the outputs
>>> print("Context Vector Shape:", context_vector.shape)
Context Vector Shape: torch.Size([5, 20])
>>> print("Attention Weights Shape:", attention_weights.shape)
Attention Weights Shape: torch.Size([5, 3])

>>> ### Example calculating attention weights based on potential difference ###
>>> # Initialize the attention module
>>> hidden_size = 20
>>> attention_module = AttentionByPotentialDifferenceModule(hidden_size)

>>> # Create dummy input tensors
>>> batch_size = 2
>>> seq_len = 5
>>> h_current = torch.randn(batch_size, hidden_size)
>>> previous_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
>>> timestep = 2  # Example for the 3rd step of the sequence (index 2)

>>> # Perform a forward pass
>>> context_vector, attention_weights = attention_module(h_current, previous_hidden_states, timestep)

>>> # Check the output shapes
>>> context_vector.shape
torch.Size([2, 20])
>>> attention_weights.shape
torch.Size([2, 5])

>>> # Example with different potential type
>>> attention_module_linear = AttentionByPotentialDifferenceModule(hidden_size, potential_type='linear')
>>> context_linear, weights_linear = attention_module_linear(h_current, previous_hidden_states, timestep)
>>> context_linear.shape
torch.Size([2, 20])
>>> weights_linear.shape
torch.Size([2, 5])

>>> ### Example LSTM cells with attention & Potential difference principle ###
>>> # Define attention parameters
>>> attention_params = {'potential_type': 'norm', 'time_resistance_lambda': 0.1, 'semantic_resistance_lambda': 1.0}

>>> # Initialize the LSTM cell with attention based on potential difference principle
>>> input_size = 10
>>> hidden_size = 20
>>> lstm_cell_attention = LSTMCellWithAttentionPD(input_size, hidden_size, attention_params)

>>> # Create dummy input tensors
>>> batch_size = 2
>>> seq_len = 5
>>> input_tensor = torch.randn(batch_size, input_size)
>>> hidden_state = torch.randn(batch_size, hidden_size)
>>> cell_state = torch.randn(batch_size, hidden_size)
>>> previous_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
>>> timestep = 3

>>> # Perform a forward pass
>>> next_hidden_state, next_cell_state, attention_weights = lstm_cell_attention(
...     input_tensor, hidden_state, cell_state, previous_hidden_states, timestep
... )

>>> # Check the output shapes
>>> next_hidden_state.shape
torch.Size([2, 20])
>>> next_cell_state.shape
torch.Size([2, 20])
>>> attention_weights.shape
torch.Size([2, 5])

>>> ### Example LSTM cells with attention & Gravitational principle by function ###
>>> import torch
>>> import torch.nn as nn
>>> import torch.nn.functional as F

>>> # Initialize the gravitational attention module
>>> hidden_size = 32

>>> # Example with base parameters for gravitational attention
>>> attention_module = GravitationalAttention(hidden_size)

>>> # Create dummy input tensors
>>> batch_size = 4
>>> seq_len = 10
>>> h_current = torch.randn(batch_size, hidden_size)
>>> previous_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
>>> timestep = 5  # Example for the 6th step of the sequence (index 5)

>>> # Perform a forward pass
>>> context_vector, attention_weights = attention_module(h_current, previous_hidden_states, timestep)

>>> # Check the output shapes
>>> context_vector.shape
torch.Size([4, 32])
>>> attention_weights.shape
torch.Size([4, 10])

>>> # Example with different weight and distance types
>>> attention_module_alt = GravitationalAttention(
...     hidden_size, gravity_constant=0.5, distance_power=1.5, weight_type='entropy', distance_type='time'
... )

>>> # Perform a forward pass with alternative parameters
>>> context_alt, weights_alt = attention_module_alt(h_current, previous_hidden_states, timestep)

>>> # Check the output shapes
>>> context_alt.shape
torch.Size([4, 32])
>>> weights_alt.shape
torch.Size([4, 10])

>>> ### Example LSTM cells with attention & Momentum principle by function ###
>>> # Initialize the momentum attention module
>>> hidden_size = 16

>>> # Example with default distance type and lambda
>>> attention_module = MomentumAttention(hidden_size)

>>> # Create dummy input tensors
>>> batch_size = 3
>>> seq_len = 8
>>> h_current = torch.randn(batch_size, hidden_size)
>>> previous_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
>>> h_prev = torch.randn(batch_size, hidden_size)

>>> # Perform a forward pass
>>> context_vector, attention_weights = attention_module(h_current, previous_hidden_states, h_prev)

>>> # Check the output shapes
>>> context_vector.shape
torch.Size([3, 16])
>>> attention_weights.shape
torch.Size([3, 8])

>>> # Example with different distance type and lambda
>>> attention_module_alt = MomentumAttention(hidden_size, distance_type='euclidean', distance_lambda=0.05)

>>> # Perform a forward pass
>>> context_alt, weights_alt = attention_module_alt(h_current, previous_hidden_states, h_prev)

>>> # Check the output shapes
>>> context_alt.shape
torch.Size([3, 16])
>>> weights_alt.shape
torch.Size([3, 8])
"""

# Imports libraries
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

### Functions version ###
# LSTM with Attention, function definition
def lstm_cell_with_attention(input_tensor, hidden_state, cell_state, weights_ih, weights_hh, bias_ih, bias_hh, attention_weights, previous_hidden_states, timestep):
    """
    Calculation of a single step of an LSTM cell with an attention mechanism.
    
    Examples:
    >>> import torch
    >>> import torch.nn.functional as F
    >>> import torch.nn as nn
    
    >>> # Setting parameters
    >>> batch_size = 2
    >>> input_size = 10
    >>> hidden_size = 15
    >>> seq_length = 5
    >>> timestep = 2  # Example for the 3rd step of the sequence (index 2)
    
    >>> # Initializing random inputs and states
    >>> input_tensor = torch.randn(batch_size, input_size)
    >>> hidden_state_prev = torch.randn(batch_size, hidden_size)
    >>> cell_state_prev = torch.randn(batch_size, hidden_size)
    >>> weights_ih = torch.randn(4 * hidden_size, input_size)
    >>> weights_hh = torch.randn(4 * hidden_size, hidden_size)
    >>> bias_ih = torch.randn(4 * hidden_size)
    >>> bias_hh = torch.randn(4 * hidden_size)
    >>> attention_weights = torch.randn(hidden_size, hidden_size)
    >>> previous_hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    
    >>> # Performing an LSTM step with attention
    >>> next_h, next_c = lstm_cell_with_attention(
    ...     input_tensor, hidden_state_prev, cell_state_prev, weights_ih, weights_hh,
    ...     bias_ih, bias_hh, attention_weights, previous_hidden_states, timestep
    ... )
    
    >>> # The shapes of the outputs
    >>> next_h.shape
    torch.Size([2, 15])
    >>> next_c.shape
    torch.Size([2, 15])
    

    Args:
        input_tensor: Input tensor (x_t), shape: (batch_size, input_size).
        hidden_state: Hidden state from the previous step (h_{t-1}), shape: (batch_size, hidden_size).
        cell_state: Cell state from the previous step (c_{t-1}), shape: (batch_size, hidden_size).
        weights_ih: Weights for the input tensor, shape: (4 * hidden_size, input_size).
        weights_hh: Weights for the hidden state, shape: (4 * hidden_size, hidden_size).
        bias_ih: Biases for the input tensor, shape: (4 * hidden_size).
        bias_hh: Biases for the hidden state, shape: (4 * hidden_size).
        attention_weights: Attention weights, shape: (hidden_size, hidden_size).
        previous_hidden_states: Hidden states from previous steps, shape: (batch_size, seq_length, hidden_size).
        timestep: Current timestep in the sequence, 0-indexed.

    Returns:
        New hidden state (h_t) and new cell state (c_t)
    """
      
    # Linear transformation (fused)
    gates = torch.matmul(input_tensor, weights_ih.t()) + bias_ih + torch.matmul(hidden_state, weights_hh.t()) + bias_hh

    # Splitting into individual gates
    input_gate = torch.sigmoid(gates[:, 0:hidden_state.shape[1]])
    forget_gate = torch.sigmoid(gates[:, hidden_state.shape[1]:2 * hidden_state.shape[1]])
    output_gate = torch.sigmoid(gates[:, 2 * hidden_state.shape[1]:3 * hidden_state.shape[1]])
    cell_gate = torch.tanh(gates[:, 3 * hidden_state.shape[1]:4 * hidden_state.shape[1]])

    # Updating the cell state
    cell_state = forget_gate * cell_state + input_gate * cell_gate

    # Attention mechanism
    # Use previous hidden states up to the current timestep
    relevant_hidden_states = previous_hidden_states[:, :timestep + 1, :]  # Shape: (batch_size, timestep + 1, hidden_size)
    # Compute attention scores: (batch_size, timestep + 1)
    attention_scores = torch.bmm(relevant_hidden_states, hidden_state.unsqueeze(2)).squeeze(2)
    attention_weights_normalized = F.softmax(attention_scores, dim=1).unsqueeze(1)  # Shape: (batch_size, 1, timestep + 1)
    context_vector = torch.bmm(attention_weights_normalized, relevant_hidden_states).squeeze(1)  # Shape: (batch_size, hidden_size)

    # Updating the hidden state with the context vector
    hidden_state = output_gate * torch.tanh(cell_state) + context_vector

    return hidden_state, cell_state


# Definition LSTM with Entropic regularization, function definition
def lstm_cell_attention_entropy_reg(input_tensor, hidden_state, cell_state, weights_ih, weights_hh, bias_ih, bias_hh, attention_weights, previous_hidden_states, lambda_entropy, timestep):
    """
    Calculation of a single step of an LSTM cell with an attention mechanism and entropic regularization.
    
    Examples:
    >>> import torch
    >>> import torch.nn.functional as F
    >>> import torch.nn as nn
    
    >>> # Setting parameters
    >>> batch_size = 2
    >>> input_size = 10
    >>> hidden_size = 15
    >>> seq_length = 5
    >>> timestep = 2  # Example for the 3rd step of the sequence (index 2)
    >>> lambda_entropy_val = 0.1  # Example value for entropic regularization coefficient
    
    >>> # Initialize random inputs and states
    >>> input_tensor = torch.randn(batch_size, input_size)
    >>> hidden_state_prev = torch.randn(batch_size, hidden_size)
    >>> cell_state_prev = torch.randn(batch_size, hidden_size)
    >>> weights_ih = torch.randn(4 * hidden_size, input_size)
    >>> weights_hh = torch.randn(4 * hidden_size, hidden_size)
    >>> bias_ih = torch.randn(4 * hidden_size)
    >>> bias_hh = torch.randn(4 * hidden_size)
    >>> attention_weights = torch.randn(hidden_size, hidden_size)
    >>> previous_hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    
    >>> # Perform LSTM step with attention and entropic regularization
    >>> next_h, next_c = lstm_cell_attention_entropy_reg(
    ...     input_tensor, hidden_state_prev, cell_state_prev, weights_ih, weights_hh,
    ...     bias_ih, bias_hh, attention_weights, previous_hidden_states, lambda_entropy_val, timestep
    ... )
    >>> next_h.shape
    torch.Size([2, 15])
    >>> next_c.shape
    torch.Size([2, 15])
      
    Args:
        input_tensor: Input tensor (x_t), shape: (batch_size, input_size).
        hidden_state: Hidden state from the previous step (h_{t-1}), shape: (batch_size, hidden_size).
        cell_state: Cell state from the previous step (c_{t-1}), shape: (batch_size, hidden_size).
        weights_ih: Weights for the input tensor, shape: (4 * hidden_size, input_size).
        weights_hh: Weights for the hidden state, shape: (4 * hidden_size, hidden_size).
        bias_ih: Biases for the input tensor, shape: (4 * hidden_size,).
        bias_hh: Biases for the hidden state, shape: (4 * hidden_size,).
        attention_weights: Attention weights (not used directly here, kept for compatibility), shape: (hidden_size, hidden_size).
        previous_hidden_states: Hidden states from previous steps, shape: (batch_size, seq_length, hidden_size).
        lambda_entropy: Coefficient of entropic regularization
        timestep: Current timestep in the sequence, 0-indexed.

    Returns:
        New hidden state (h_t) and new cell state (c_t)
    """
    # Linear transformation (fused)
    gates = torch.matmul(input_tensor, weights_ih.t()) + bias_ih + torch.matmul(hidden_state, weights_hh.t()) + bias_hh

    # Splitting into individual gates
    input_gate = torch.sigmoid(gates[:, 0:hidden_state.shape[1]])
    forget_gate = torch.sigmoid(gates[:, hidden_state.shape[1]:2 * hidden_state.shape[1]])
    output_gate = torch.sigmoid(gates[:, 2 * hidden_state.shape[1]:3 * hidden_state.shape[1]])
    cell_gate = torch.tanh(gates[:, 3 * hidden_state.shape[1]:4 * hidden_state.shape[1]])

    # Updating the cell state
    cell_state = forget_gate * cell_state + input_gate * cell_gate

    # Attention mechanism
    relevant_hidden_states = previous_hidden_states[:, :timestep + 1, :]  # Shape: (batch_size, timestep + 1, hidden_size)
    if timestep > 0:  # Only apply attention if there are previous states
        # Compute attention scores over the sequence
        attention_scores = torch.bmm(relevant_hidden_states, hidden_state.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, timestep + 1)
        attention_weights_normalized = F.softmax(attention_scores, dim=1)  # Shape: (batch_size, timestep + 1)
        context_vector = torch.bmm(attention_weights_normalized.unsqueeze(1), relevant_hidden_states).squeeze(1)  # Shape: (batch_size, hidden_size)

        # Calculate entropy of attention weights
        entropy = -torch.sum(attention_weights_normalized * torch.log(attention_weights_normalized + 1e-8), dim=1).mean()
    else:
        context_vector = torch.zeros_like(hidden_state, device=hidden_state.device)  # No attention at t=0
        entropy = torch.tensor(0.0, device=hidden_state.device)

    # Update hidden state with context vector and entropy regularization
    hidden_state = output_gate * torch.tanh(cell_state) + context_vector - lambda_entropy * entropy

    return hidden_state, cell_state


# Definition LSTM with Time Decay, function definition
def lstm_cell_attention_time_decay(input_tensor, hidden_state, cell_state, weights_ih, weights_hh, bias_ih, bias_hh, attention_weights, previous_hidden_states, timestep, time_decay):
    """
    Calculation of a single step of an LSTM cell with an attention mechanism and time decay.
    
    Examples:
    >>> import torch
    >>> import torch.nn.functional as F
    >>> import torch.nn as nn
   
    >>> # Setting parameters
    >>> batch_size = 2
    >>> input_size = 10
    >>> hidden_size = 15
    >>> seq_length = 5
    >>> timestep = 2  # Example for the 3rd step of the sequence (index 2)
    >>> time_decay_val = 0.1  # Example value for time decay coefficient
    
    >>> # Initialize random inputs and states
    >>> input_tensor = torch.randn(batch_size, input_size)
    >>> hidden_state_prev = torch.randn(batch_size, hidden_size)
    >>> cell_state_prev = torch.randn(batch_size, hidden_size)
    >>> weights_ih = torch.randn(4 * hidden_size, input_size)
    >>> weights_hh = torch.randn(4 * hidden_size, hidden_size)
    >>> bias_ih = torch.randn(4 * hidden_size)
    >>> bias_hh = torch.randn(4 * hidden_size)
    >>> attention_weights_dummy = torch.randn(hidden_size, hidden_size)
    >>> previous_hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    
    >>> # Perform LSTM step with attention and temporal decay
    >>> next_h, next_c = lstm_cell_attention_time_decay(input_tensor, hidden_state_prev, cell_state_prev, weights_ih, weights_hh, bias_ih, bias_hh, attention_weights_dummy, previous_hidden_states, timestep, time_decay_val)
    >>> next_h.shape
    torch.Size([2, 15])
    >>> next_c.shape
    torch.Size([2, 15])

    Args:
        input_tensor: Input tensor (x_t), shape: (batch_size, input_size).
        hidden_state: Hidden state from the previous step (h_{t-1}), shape: (batch_size, hidden_size).
        cell_state: Cell state from the previous step (c_{t-1}), shape: (batch_size, hidden_size).
        weights_ih: Weights for the input tensor, shape: (4 * hidden_size, input_size).
        weights_hh: Weights for the hidden state, shape: (4 * hidden_size, hidden_size).
        bias_ih: Biases for the input tensor, shape: (4 * hidden_size,).
        bias_hh: Biases for the hidden state, shape: (4 * hidden_size,).
        attention_weights: Attention weights (not used directly here, kept for compatibility), shape: (hidden_size, hidden_size).
        previous_hidden_states: Hidden states from previous steps, shape: (batch_size, seq_length, hidden_size).
        timestep: Current timestep in the sequence, 0-indexed.
        time_decay: Coefficient of time decay

    Returns:
        New hidden state (h_t) and new cell state (c_t)
    """

    # Linear transformation (fused)
    gates = torch.matmul(input_tensor, weights_ih.t()) + bias_ih + torch.matmul(hidden_state, weights_hh.t()) + bias_hh

    # Splitting into individual gates
    input_gate = torch.sigmoid(gates[:, 0:hidden_state.shape[1]])
    forget_gate = torch.sigmoid(gates[:, hidden_state.shape[1]:2 * hidden_state.shape[1]])
    output_gate = torch.sigmoid(gates[:, 2 * hidden_state.shape[1]:3 * hidden_state.shape[1]])
    cell_gate = torch.tanh(gates[:, 3 * hidden_state.shape[1]:4 * hidden_state.shape[1]])

    # Updating the cell state
    cell_state = forget_gate * cell_state + input_gate * cell_gate

    # Attention
    relevant_hidden_states = previous_hidden_states[:, :timestep + 1, :]
    if timestep > 0:
        # Compute attention scores
        attention_scores = torch.bmm(relevant_hidden_states, hidden_state.unsqueeze(2)).squeeze(2)
        attention_weights_unnormalized = torch.exp(attention_scores) # Using exponential for non-negative weights

        # Apply time decay to attention weights
        time_weights = torch.exp(-time_decay * torch.arange(timestep + 1).float().to(input_tensor.device))
        time_weights = time_weights.unsqueeze(0) # Add batch dimension
        attention_weights_with_decay = attention_weights_unnormalized * time_weights

        # Normalize attention weights with decay
        attention_weights_normalized = attention_weights_with_decay / torch.sum(attention_weights_with_decay, dim=1, keepdim=True)

        # Compute context vector
        context_vector = torch.bmm(attention_weights_normalized.unsqueeze(1), relevant_hidden_states).squeeze(1)
    else:
        context_vector = torch.zeros_like(hidden_state, device=hidden_state.device)

    # Updating the hidden state with the context vector
    hidden_state = output_gate * torch.tanh(cell_state) + context_vector

    return hidden_state, cell_state


# Definition LSTM with Entropic regularization & Time factor & randomness, function definition
def lstm_cell_attention_mixed(input_tensor, hidden_state, cell_state, weights_ih, weights_hh, bias_ih, bias_hh, attention_weights, previous_hidden_states, timestep, lambda_entropy, time_decay, random_prob):
    """
    Calculation of a single step of an LSTM cell with an attention mechanism, entropic regularization,
    time decay, and a random element.
    
    Excercises:
    >>> import torch
    >>> import torch.nn.functional as F
    >>> import torch.nn as nn
    
    >>> # Setting parameters
    >>> batch_size = 2
    >>> input_size = 10
    >>> hidden_size = 15
    >>> seq_length = 5
    >>> timestep = 2  # Example for the 3rd step of the sequence (index 2)
    >>> lambda_entropy_val = 0.01  # Example value for entropic regularization coefficient
    >>> time_decay_val = 0.1  # Example value for time decay coefficient
    >>> random_prob_val = 0.3
    
    >>> # Initialize random inputs and states
    >>> input_tensor = torch.randn(batch_size, input_size)
    >>> hidden_state_prev = torch.randn(batch_size, hidden_size)
    >>> cell_state_prev = torch.randn(batch_size, hidden_size)
    >>> weights_ih = torch.randn(4 * hidden_size, input_size)
    >>> weights_hh = torch.randn(4 * hidden_size, hidden_size)
    >>> bias_ih = torch.randn(4 * hidden_size)
    >>> bias_hh = torch.randn(4 * hidden_size)
    >>> attention_weights_dummy = torch.randn(hidden_size, hidden_size) # Not used directly
    >>> previous_hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    
    >>> # Performing the LSTM step with complex attention
    >>> # It is difficult to test for randomness deterministically, so we focus on the shape of the output
    >>> next_h, next_c = lstm_cell_attention_mixed(
    ...     input_tensor, hidden_state_prev, cell_state_prev, weights_ih, weights_hh,
    ...     bias_ih, bias_hh, attention_weights_dummy, previous_hidden_states,
    ...     timestep, lambda_entropy_val, time_decay_val, random_prob_val
    ... )
    
    >>> next_h.shape
    torch.Size([2, 15])
    >>> next_c.shape
    torch.Size([2, 15])

    Args:
        input_tensor: Input tensor (x_t), shape: (batch_size, input_size).
        hidden_state: Hidden state from the previous step (h_{t-1}), shape: (batch_size, hidden_size).
        cell_state: Cell state from the previous step (c_{t-1}), shape: (batch_size, hidden_size).
        weights_ih: Weights for the input tensor, shape: (4 * hidden_size, input_size).
        weights_hh: Weights for the hidden state, shape: (4 * hidden_size, hidden_size).
        bias_ih: Biases for the input tensor, shape: (4 * hidden_size,).
        bias_hh: Biases for the hidden state, shape: (4 * hidden_size,).
        attention_weights: Attention weights (not used directly here, kept for compatibility), shape: (hidden_size, hidden_size).
        previous_hidden_states: Hidden states from previous steps, shape: (batch_size, seq_length, hidden_size).
        timestep: Current timestep in the sequence, 0-indexed.
        lambda_entropy: Coefficient of entropic regularization, between 0 and 1
        time_decay: Coefficient of time decay
        random_prob: Probability of applying random zeroing of attention weights, between 0 and 1

    Returns:
        New hidden state (h_t) and new cell state (c_t)
    """
    # Linear transformation (fused)
    gates = torch.matmul(input_tensor, weights_ih.t()) + bias_ih + torch.matmul(hidden_state, weights_hh.t()) + bias_hh

    # Splitting into individual gates
    input_gate = torch.sigmoid(gates[:, 0:hidden_state.shape[1]])
    forget_gate = torch.sigmoid(gates[:, hidden_state.shape[1]:2 * hidden_state.shape[1]])
    output_gate = torch.sigmoid(gates[:, 2 * hidden_state.shape[1]:3 * hidden_state.shape[1]])
    cell_gate = torch.tanh(gates[:, 3 * hidden_state.shape[1]:4 * hidden_state.shape[1]])

    # Updating the cell state
    cell_state = forget_gate * cell_state + input_gate * cell_gate

    # Attention
    relevant_hidden_states = previous_hidden_states[:, :timestep + 1, :]
    if timestep > 0:
        # Compute attention scores
        attention_scores = torch.bmm(relevant_hidden_states, hidden_state.unsqueeze(2)).squeeze(2)
        attention_weights_unnormalized = torch.exp(attention_scores)

        # Apply time decay
        time_weights = torch.exp(-time_decay * torch.arange(timestep + 1).float().to(input_tensor.device))
        time_weights = time_weights.unsqueeze(0)
        attention_weights_with_decay = attention_weights_unnormalized * time_weights

        # Normalize attention weights
        attention_weights_normalized_base = attention_weights_with_decay / torch.sum(attention_weights_with_decay, dim=1, keepdim=True)

        # Apply randomness
        if random.random() < random_prob:
            random_mask = torch.rand(attention_weights_normalized_base.shape, device=input_tensor.device) < 0.5
            attention_weights_normalized = attention_weights_normalized_base.masked_fill(random_mask, 0)
            attention_weights_normalized = attention_weights_normalized / (torch.sum(attention_weights_normalized, dim=1, keepdim=True) + 1e-8) # Re-normalize
        else:
            attention_weights_normalized = attention_weights_normalized_base

        # Compute context vector
        context_vector = torch.bmm(attention_weights_normalized.unsqueeze(1), relevant_hidden_states).squeeze(1)

        # Calculate entropy of attention weights
        entropy = -torch.sum(attention_weights_normalized * torch.log(attention_weights_normalized + 1e-8), dim=1).mean()
    else:
        context_vector = torch.zeros_like(hidden_state, device=hidden_state.device)
        entropy = torch.tensor(0.0, device=hidden_state.device)

    # Updating the hidden state with the context vector and entropic regularization
    hidden_state = output_gate * torch.tanh(cell_state) + context_vector - lambda_entropy * entropy

    return hidden_state, cell_state


# Definition LSTM with Potential difference principle, function definition
def attention_by_potential_difference(h_current, previous_hidden_states, potential_func, time_resistance_func, semantic_resistance_func, timestep):
    """
    Calculation of attention weights based on the principle of potential difference.
    
    Examples:
    >>> import torch
    >>> import torch.nn.functional as F
    >>> import torch.nn as nn
    
    >>> # Define dummy functions for potential and resistances
    >>> def dummy_potential(h):
    ...     return torch.norm(h, dim=-1, keepdim=True)
    
    >>> def dummy_time_resistance(time_diff):
    ...     return torch.exp(0.1 * time_diff)
    
    >>> def dummy_semantic_resistance(h_current, h_past):
    ...     cos_sim = F.cosine_similarity(h_current, h_past, dim=-1, eps=1e-8)
    ...     return torch.exp(1.0 * (1 - cos_sim.unsqueeze(-1)))
    
    >>> # Setting parameters
    >>> batch_size = 2
    >>> hidden_size = 10
    >>> seq_length = 5
    >>> timestep = 2  # Example for the 3rd step of the sequence (index 2)
    
    >>> # Initialize random inputs and states
    >>> h_current = torch.randn(batch_size, hidden_size)
    >>> previous_hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    
    >>> # Calculation of attention weights
    >>> attention_weights = attention_by_potential_difference(
    ...     h_current, previous_hidden_states, dummy_potential, dummy_time_resistance, dummy_semantic_resistance, timestep
    ... )
    
    >>> attention_weights.shape
    torch.Size([2, 5])
    >>> # Kontrola, zda jsou váhy normalizované (součet řádků by měl být blízko 1)
    >>> torch.allclose(torch.sum(attention_weights, dim=1), torch.ones(batch_size), atol=1e-6)
    True
    
    Args:
        h_current (torch.Tensor): Current hidden state (h_{t-1}), shape: [batch_size, hidden_size]
        previous_hidden_states (torch.Tensor): Hidden states from previous steps,
                                              shape: [batch_size, seq_len, hidden_size]
        potential_func (callable): Function to calculate the potential of a hidden state,
                                   takes a tensor [..., hidden_size] and returns [..., 1]
        time_resistance_func (callable): Function to calculate the time resistance,
                                        takes the time difference and returns a tensor [..., 1]
        semantic_resistance_func (callable): Function to calculate the semantic resistance,
                                            takes two hidden state tensors
                                            [..., hidden_size] and returns [..., 1]
        timestep (int): Current timestep to mask future states

    Returns:
        torch.Tensor: Normalized attention weights, shape: [batch_size, seq_len]
    """
    batch_size, seq_len, hidden_size = previous_hidden_states.size()

    potential_current = potential_func(h_current)  # [batch_size, 1]
    potential_past = potential_func(previous_hidden_states)  # [batch_size, seq_len, 1]

    # Calculation of time resistance
    time_diff = torch.arange(seq_len - 1, -1, -1).float().to(h_current.device).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
    time_resistance = time_resistance_func(time_diff)  # [1, seq_len, 1]

    # Calculation of semantic resistance
    semantic_resistance = semantic_resistance_func(h_current.unsqueeze(1), previous_hidden_states)  # [batch_size, seq_len, 1]

    # Total resistance
    resistance = time_resistance + semantic_resistance + 1e-8  # [batch_size, seq_len, 1]

    # Potential difference
    potential_difference = torch.abs(potential_current.unsqueeze(1) - potential_past)  # [batch_size, seq_len, 1]

    # Information flow
    flow = potential_difference / resistance  # [batch_size, seq_len, 1]

    # Mask future timesteps (after current timestep)
    mask = torch.ones(batch_size, seq_len, 1, device=h_current.device)
    mask[:, timestep + 1 :, :] = -float('inf')  # Mask future steps
    flow = flow + mask  # Apply mask to flow before softmax

    # Normalization to attention weights
    attention_weights = F.softmax(flow, dim=1).squeeze(-1)  # [batch_size, seq_len]

    return attention_weights


# Definition Gravitational Attention, function definition
def gravitational_attention_function(h_current, previous_hidden_states, weight_func, distance_func, gravity_constant=1.0, distance_power=2.0, timestep=None):
    """
    Calculation of attention weights based on an analogy with the law of gravity.
    
    Examples:
    >>> import torch
    >>> import torch.nn.functional as F
    >>> import torch.nn as nn
    
    >>> # Define dummy functions for weight and distance
    >>> def dummy_weight(h):
    ...     return torch.norm(h, dim=-1, keepdim=True)
    
    >>> def dummy_distance(h_current, h_past):
    ...     return torch.cdist(h_current.unsqueeze(1), h_past, p=2).squeeze(1)
    
    >>> # Setting parameters
    >>> batch_size = 2
    >>> seq_len = 5
    >>> hidden_size = 10
    >>> h_current = torch.randn(batch_size, hidden_size)
    >>> previous_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    >>> # Calculation of attention weights without temporal masking
    >>> attention_weights = gravitational_attention_function(
    ...     h_current, previous_hidden_states, dummy_weight, dummy_distance, gravity_constant=0.5, distance_power=2.0
    ... )
    
    >>> attention_weights.shape
    torch.Size([2, 5, 1])
    
    >>> # Check if the weights are normalized (row sum should be close to 1)
    >>> torch.allclose(torch.sum(attention_weights, dim=1), torch.ones(batch_size, 1, 1), atol=1e-6)
    True
    
    >>> # Calculating attention weights with temporal masking
    >>> timestep = 2  # Example for the 3rd step of the sequence (index 2)
    >>> attention_weights_masked = gravitational_attention_function(
    ...     h_current, previous_hidden_states, dummy_weight, dummy_distance, gravity_constant=0.5, distance_power=2.0, timestep=timestep
    ... )
    >>> attention_weights_masked.shape
    torch.Size([2, 5, 1])
    
    >>> # Check if the weights for future steps (after timestep=2) are very small (close to 0 after softmax)
    >>> future_weights = attention_weights_masked[:, timestep + 1 :, :]
    >>> torch.all(future_weights < 1e-5)
    tensor(True)
    
    Args:
        h_current (torch.Tensor): Current hidden state (h_{t-1}), shape: [batch_size, hidden_size]
        previous_hidden_states (torch.Tensor): Hidden states from previous steps,
                                               shape: [batch_size, seq_len, hidden_size]
        weight_func (callable): Function to calculate the "weight" of a hidden state,
                                takes a tensor [..., hidden_size] and returns [..., 1]
        distance_func (callable): Function to calculate the "distance" between two tensors
                                  of hidden states [..., hidden_size], returns [..., 1] or [...]
        gravity_constant (float, torch.Tensor): "Gravitational constant".
        distance_power (float, torch.Tensor): Exponent for the distance.
        timestep (int, optional): Current timestep for causal masking.

    Returns:
        torch.Tensor: Normalized attention weights, shape: [batch_size, seq_len, 1]
    """
    batch_size, seq_len, hidden_size = previous_hidden_states.size()

    weight_current = weight_func(h_current)  # [batch_size, 1]
    weight_past = weight_func(previous_hidden_states)  # [batch_size, seq_len, 1]

    # Calculation of distances
    distance = distance_func(h_current, previous_hidden_states)  # [batch_size, seq_len]

    # Ensure correct dimensions for calculation
    if distance.ndim == 2:
        distance = distance.unsqueeze(-1)  # [batch_size, seq_len, 1]
    else:
        raise ValueError(f"Unexpected distance tensor shape: {distance.shape}")

    # Calculation of "gravitational force"
    gravity_force = gravity_constant * (weight_current.unsqueeze(1) * weight_past) / (distance.pow(distance_power) + 1e-8)  # [batch_size, seq_len, 1]

    # Apply causal mask if timestep is provided
    if timestep is not None:
        mask = torch.ones(batch_size, seq_len, 1, device=h_current.device)
        mask[:, timestep + 1 :, :] = -float('inf')  # Mask future steps
        gravity_force = gravity_force + mask

    # Normalization to attention weights
    attention_weights = F.softmax(gravity_force, dim=1)  # [batch_size, seq_len, 1]

    return attention_weights


# Definition Momentum Attention, function definition
def momentum_attention_function(h_current, previous_hidden_states, h_prev, weight_func, distance_func, distance_lambda=0.1):
    """
    Calculation of attention weights based on the principle of momentum.
    
    Examples:
    >>> import torch
    >>> import torch.nn.functional as F
    >>> import torch.nn as nn
    
    >>> # Define dummy functions for weight and distance
    >>> def dummy_weight(h):
    ...     return torch.norm(h, dim=-1, keepdim=True)
    
    >>> def dummy_distance(h_current, h_past):
    ...     return torch.cdist(h_current.unsqueeze(1), h_past, p=2).squeeze(1).squeeze(-1)
    
    >>> # Setting parameters
    >>> batch_size = 2
    >>> seq_len = 5
    >>> hidden_size = 10
    >>> h_current = torch.randn(batch_size, hidden_size)
    >>> previous_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    >>> h_prev = torch.randn(batch_size, hidden_size)
    
    >>> # Calculating attention weights
    >>> attention_weights = momentum_attention_function(
    ...     h_current, previous_hidden_states, h_prev, dummy_weight, dummy_distance, distance_lambda=0.05
    ... )
    >>> attention_weights.shape
    torch.Size([2, 5, 1])
    
    >>> # Check if the weights are normalized (row sum should be close to 1)
    >>> torch.allclose(torch.sum(attention_weights, dim=1), torch.ones(batch_size, 1, 1), atol=1e-6)
    True

    Args:
        h_current (torch.Tensor): Current hidden state (h_{t-1}), shape: [batch_size, hidden_size]
        previous_hidden_states (torch.Tensor): Hidden states from previous steps,
                                               shape: [batch_size, seq_len, hidden_size]
        h_prev (torch.Tensor): Previous hidden state (h_{t-2}), shape: [batch_size, hidden_size]
        weight_func (callable): Function to calculate the "weight" of a hidden state,
                                takes a tensor [..., hidden_size] and returns [..., 1]
        distance_func (callable): Function to calculate the "distance" between two tensors
                                  of hidden states [..., hidden_size], returns [..., 1] or [...]
        distance_lambda (float, torch.Tensor): Coefficient for the influence of distance.

    Returns:
        torch.Tensor: Normalized attention weights, shape: [batch_size, seq_len, 1]
    """
    batch_size, seq_len, hidden_size = previous_hidden_states.size()

    weight_current = weight_func(h_current) # [batch_size, 1]
    weight_past = weight_func(previous_hidden_states) # [batch_size, seq_len, 1]

    # Calculation of "velocity"
    velocity = (h_current - h_prev.detach()).unsqueeze(1) # [batch_size, 1, hidden_size]

    # Calculation of "momentum" (direction)
    momentum = weight_current.unsqueeze(1) * velocity # [batch_size, 1, hidden_size]

    # Calculation of "alignment" with previous states
    alignment = torch.bmm(momentum, previous_hidden_states.transpose(1, 2)).squeeze(1) # [batch_size, seq_len]

    # Calculation of distance
    distance = distance_func(h_current, previous_hidden_states).squeeze(-1) # [batch_size, seq_len]

    # Calculation of score
    score = alignment + weight_past.squeeze(-1) - distance_lambda * distance

    # Normalization to attention weights
    attention_weights = F.softmax(score, dim=1).unsqueeze(-1) # [batch_size, seq_len, 1]

    return attention_weights
     

### Additional classes ###
# Encapsulating attention calculation logic into a class, class definition
class AttentionByPotentialDifferenceModule(nn.Module):
    """
    Attention module calculating attention weights based on potential difference.

    This module encapsulates the logic for calculating attention weights using
    potential functions and resistance functions (time and semantic).
    
    Args:
        hidden_size (int): The number of features in the hidden state.
        potential_type (str, optional): Type of potential function to use. Options are 'norm', 'linear', 'tanh'. Defaults to 'norm'.
        time_resistance_lambda (float, optional): Lambda parameter for the time resistance function. Defaults to 0.1.
        semantic_resistance_lambda (float, optional): Lambda parameter for the semantic resistance function. Defaults to 1.0.

    """
    def __init__(self, hidden_size, potential_type='norm', time_resistance_lambda=0.1, semantic_resistance_lambda=1.0):
        """
        Initializes the AttentionByPotentialDifferenceModule.

        Args:
            hidden_size (int): The number of features in the hidden state.
            potential_type (str, optional): Type of potential function ('norm', 'linear', 'tanh'). Defaults to 'norm'.
            time_resistance_lambda (float, optional): Scaling factor for time resistance. Defaults to 0.1.
            semantic_resistance_lambda (float, optional): Scaling factor for semantic resistance. Defaults to 1.0.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.potential_type = potential_type
        self.time_resistance_lambda = nn.Parameter(torch.tensor(time_resistance_lambda))
        self.semantic_resistance_lambda = nn.Parameter(torch.tensor(semantic_resistance_lambda))
        self.wp = nn.Linear(hidden_size, 1)

    def potential(self, h):
        """
        Calculates the potential of a hidden state.

        Args:
            h (torch.Tensor): Hidden state tensor, shape: [..., hidden_size]

        Returns:
            torch.Tensor: Potential value, shape: [..., 1]

        Raises:
            ValueError: If an unknown potential type is specified.

        Examples:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> import torch.nn as nn
        
        >>> # Setting parameters
        >>> hidden_size = 10
        >>> batch_size = 2
        >>> h = torch.randn(batch_size, hidden_size)
        
        >>> # Sample 1
        >>> module = AttentionByPotentialDifferenceModule(hidden_size)
        >>> module.potential(h).shape
        torch.Size([2, 1])
        >>> # Sample 2
        >>> module = AttentionByPotentialDifferenceModule(hidden_size, potential_type='linear')
        >>> module.potential(h).shape
        torch.Size([2, 1])
        """
        if self.potential_type == 'norm':
            return torch.norm(h, dim=-1, keepdim=True)
        elif self.potential_type == 'linear':
            return self.wp(h)
        elif self.potential_type == 'tanh':
            return torch.tanh(self.wp(h))
        else:
            raise ValueError(f"Unknown potential type: {self.potential_type}")

    def time_resistance(self, time_diff):
        """
        Calculates the time resistance based on the time difference.

        Args:
            time_diff (torch.Tensor): Tensor of time differences, shape: [..., 1]

        Returns:
            torch.Tensor: Time resistance value, shape: [..., 1]

        Examples:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> import torch.nn as nn
        
        >>> # Setting parameters
        >>> hidden_size = 10
        
        >>> # Initializing module and calculate time resistance
        >>> module = AttentionByPotentialDifferenceModule(hidden_size)
        >>> time_diff = torch.arange(5).float().unsqueeze(-1)
        >>> module.time_resistance(time_diff).shape
        torch.Size([5, 1])
        """
        return torch.exp(self.time_resistance_lambda * time_diff)

    def semantic_resistance(self, h_current, h_past):
        """
        Calculates the semantic resistance between the current and past hidden states.

        Args:
            h_current (torch.Tensor): Current hidden state, shape: [..., hidden_size]
            h_past (torch.Tensor): Past hidden state, shape: [..., hidden_size]

        Returns:
            torch.Tensor: Semantic resistance value, shape: [..., 1]

        Examples:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> import torch.nn as nn
        
        >>> # Setting parameters
        >>> hidden_size = 10
        >>> batch_size = 2
        >>> h_current = torch.randn(batch_size, hidden_size)
        >>> h_past = torch.randn(batch_size, hidden_size)
        
        >>> # Initializing module and calculate semantic resistance
        >>> module = AttentionByPotentialDifferenceModule(hidden_size)
        >>> module.semantic_resistance(h_current, h_past).shape
        torch.Size([2, 1])
        """
        cos_sim = F.cosine_similarity(h_current, h_past, dim=-1, eps=1e-8)
        return torch.exp(self.semantic_resistance_lambda * (1 - cos_sim.unsqueeze(-1)))

    def forward(self, h_current, previous_hidden_states, timestep):
        """
        Forward pass of the AttentionByPotentialDifferenceModule.

        Args:
            h_current (torch.Tensor): Current hidden state, shape: [batch_size, hidden_size]
            previous_hidden_states (torch.Tensor): Previous hidden states, shape: [batch_size, seq_len, hidden_size]
            timestep (int): Current timestep to mask future states

        Returns:
            tuple: A tuple containing:
                - context_vector (torch.Tensor): Context vector, shape: [batch_size, hidden_size]
                - attention_weights (torch.Tensor): Attention weights, shape: [batch_size, seq_len]

        Examples:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> import torch.nn as nn
        
        >>> # Setting parameters
        >>> hidden_size = 20
        >>> batch_size = 2
        >>> seq_len = 5
        >>> h_current = torch.randn(batch_size, hidden_size)
        >>> previous_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        >>> timestep = 2
        
        >>> # Initializing module and calculate context vector and attention weights
        >>> attention_module = AttentionByPotentialDifferenceModule(hidden_size)
        >>> context, weights = attention_module(h_current, previous_hidden_states, timestep)
        >>> context.shape
        torch.Size([2, 20])
        >>> weights.shape
        torch.Size([2, 5])
        """
        attention_weights = attention_by_potential_difference(
            h_current,
            previous_hidden_states,
            self.potential,
            self.time_resistance,
            self.semantic_resistance,
            timestep
        )
        context_vector = torch.bmm(attention_weights.unsqueeze(1), previous_hidden_states).squeeze(1)  # [batch_size, hidden_size]
        return context_vector, attention_weights

# LSTM cells with attention & Potential difference principle, class definition
class LSTMCellWithAttentionPD(nn.Module):
    """
    A single LSTM cell with an attention mechanism based on potential difference.

    This cell computes the next hidden state and cell state by incorporating
    a context vector derived from previous hidden states using the
    AttentionByPotentialDifferenceModule.

    Args:
        input_size (int): The number of expected features in the input x.
        hidden_size (int): The number of features in the hidden state h.
        attention_params (dict): Dictionary of parameters to be passed to the AttentionByPotentialDifferenceModule.
    """
    def __init__(self, input_size, hidden_size, attention_params):
        """
        Initializes the LSTMCellWithAttentionPD.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            attention_params (dict): Dictionary of parameters for the AttentionByPotentialDifferenceModule.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size + hidden_size, hidden_size)  # Input + Context
        self.attention = AttentionByPotentialDifferenceModule(hidden_size, **attention_params)

    def forward(self, input_tensor, hidden_state, cell_state, previous_hidden_states, timestep):
        """
        Forward pass of the LSTMCellWithAttentionPD.

        Args:
            input_tensor (torch.Tensor): Input tensor for the current timestep, shape: [batch_size, input_size].
            hidden_state (torch.Tensor): Hidden state from the previous timestep, shape: [batch_size, hidden_size].
            cell_state (torch.Tensor): Cell state from the previous timestep, shape: [batch_size, hidden_size].
            previous_hidden_states (torch.Tensor): Hidden states from all previous timesteps, shape: [batch_size, seq_len, hidden_size].
            timestep (int): The current timestep in the sequence (0-indexed).

        Returns:
            tuple: A tuple containing:
                - next_hidden_state (torch.Tensor): The next hidden state, shape: [batch_size, hidden_size].
                - next_cell_state (torch.Tensor): The next cell state, shape: [batch_size, hidden_size].
                - attention_weights (torch.Tensor): The attention weights over the previous hidden states,
                  shape: [batch_size, seq_len].
        """
        # Compute context vector using attention
        context_vector, attention_weights = self.attention(hidden_state, previous_hidden_states, timestep)

        # Concatenate input and context vector
        lstm_input = torch.cat((input_tensor, context_vector), dim=1)

        # Update LSTM state
        hidden_state, cell_state = self.lstm_cell(lstm_input, (hidden_state, cell_state))

        return hidden_state, cell_state, attention_weights


# LSTM cells with attention & Gravitational principle, class definition
class GravitationalAttention(nn.Module):
    """
    Attention module calculating attention weights based on a gravitational analogy.

    This module computes attention weights by considering the "mass" (weight) of
    previous hidden states and their "distance" from the current hidden state.
    The "gravitational force" analogy determines the attention strength.

    Args:
        hidden_size (int): The number of features in the hidden state.
        gravity_constant (float, optional): The gravitational constant. Defaults to 1.0.
        distance_power (float, optional): The exponent for the distance calculation. Defaults to 2.0.
        weight_type (str, optional): Type of function to calculate the weight of a hidden state. Options are 'norm' (L2 norm) and 'entropy'. Defaults to 'norm'.
        distance_type (str, optional): Type of function to calculate the distance between hidden states. Options are 'euclidean' and 'time' (temporal distance). Defaults to 'euclidean'.
    """
    def __init__(self, hidden_size, gravity_constant=1.0, distance_power=2.0, weight_type='norm', distance_type='euclidean'):
        """
        Initializes the GravitationalAttention module.

        Args:
            hidden_size (int): The number of features in the hidden state.
            gravity_constant (float, optional): The gravitational constant. Defaults to 1.0.
            distance_power (float, optional): The exponent for the distance calculation. Defaults to 2.0.
            weight_type (str, optional): Type of function to calculate the weight ('norm', 'entropy'). Defaults to 'norm'.
            distance_type (str, optional): Type of function to calculate the distance ('euclidean', 'time'). Defaults to 'euclidean'.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.gravity_constant = nn.Parameter(torch.tensor(float(gravity_constant)))
        self.distance_power = nn.Parameter(torch.tensor(float(distance_power)))
        self.weight_type = weight_type
        self.distance_type = distance_type

    def calculate_weight(self, h):
        """
        Calculates the "weight" of a hidden state.

        Args:
            h (torch.Tensor): Hidden state tensor, shape: [..., hidden_size]

        Returns:
            torch.Tensor: Weight value, shape: [..., 1]

        Raises:
            ValueError: If an unknown weight type is specified.

        Examples:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> import torch.nn as nn
        
        >>> # Setting parameters
        >>> hidden_size = 16
        >>> batch_size = 2
        >>> h = torch.randn(batch_size, hidden_size)
        
        >>> # Sample 1, Initialize gravitational attention with default weight type (norm)
        >>> attention = GravitationalAttention(hidden_size)
        >>> attention.calculate_weight(h).shape
        torch.Size([2, 1])
        >>> # Sample 2, Initialize gravitational attention with entropy weight type
        >>> attention = GravitationalAttention(hidden_size, weight_type='entropy')
        >>> attention.calculate_weight(h).shape
        torch.Size([2, 1])
        """
        if self.weight_type == 'norm':
            return torch.norm(h, dim=-1, keepdim=True)
        elif self.weight_type == 'entropy':
            p = F.softmax(h, dim=-1)
            return -(p * torch.log(p + 1e-8)).sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown weight type: {self.weight_type}")

    def calculate_distance(self, h_current, previous_hidden_states):
        """
        Calculates the "distance" between the current and previous hidden states.

        Args:
            h_current (torch.Tensor): Current hidden state, shape: [batch_size, hidden_size]
            previous_hidden_states (torch.Tensor): Previous hidden states, shape: [batch_size, seq_len, hidden_size]

        Returns:
            torch.Tensor: Distance values, shape: [batch_size, seq_len]

        Raises:
            ValueError: If an unknown distance type is specified.

        Examples:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> import torch.nn as nn
        
        >>> # Setting parameters
        >>> hidden_size = 16
        >>> batch_size = 2
        >>> seq_len = 5
        >>> h_current = torch.randn(batch_size, hidden_size)
        >>> previous_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        >>> # Sample 1, Initialize gravitational attention with default distance type (euclidean)
        >>> attention = GravitationalAttention(hidden_size)
        >>> attention.calculate_distance(h_current, previous_hidden_states).shape
        torch.Size([2, 5])
        >>> # Sample 2, Initialize gravitational attention with time distance type
        >>> attention = GravitationalAttention(hidden_size, distance_type='time')
        >>> attention.calculate_distance(h_current, previous_hidden_states).shape
        torch.Size([1, 5])
        """
        if self.distance_type == 'euclidean':
            # Compute Euclidean distance between h_current and each previous state
            diff = h_current.unsqueeze(1) - previous_hidden_states  # [batch_size, seq_len, hidden_size]
            return torch.norm(diff, dim=-1) + 1e-8  # [batch_size, seq_len]
        elif self.distance_type == 'time':
            batch_size, seq_len, _ = previous_hidden_states.size()
            return torch.arange(seq_len, 0, -1).float().to(h_current.device).unsqueeze(0) + 1e-8  # [1, seq_len]
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

    def forward(self, h_current, previous_hidden_states, timestep=None):
        """
        Forward pass of the GravitationalAttention module.

        Args:
            h_current (torch.Tensor): Current hidden state, shape: [batch_size, hidden_size]
            previous_hidden_states (torch.Tensor): Previous hidden states, shape: [batch_size, seq_len, hidden_size]
            timestep (int, optional): Current timestep for causal masking.

        Returns:
            tuple: A tuple containing:
                - context_vector (torch.Tensor): Context vector, shape: [batch_size, hidden_size]
                - attention_weights (torch.Tensor): Attention weights, shape: [batch_size, seq_len]

        Examples:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> import torch.nn as nn
        
        >>> # Setting parameters
        >>> hidden_size = 32
        >>> batch_size = 4
        >>> seq_len = 10
        >>> h_current = torch.randn(batch_size, hidden_size)
        >>> previous_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        >>> # Sample 1, Initialize gravitational attention with default parameters
        >>> attention_module = GravitationalAttention(hidden_size)
        >>> context, weights = attention_module(h_current, previous_hidden_states, 5)
        >>> context.shape
        torch.Size([4, 32])
        >>> weights.shape
        torch.Size([4, 10])
        >>> # Sample 2, Initialize gravitational attention with time distance type
        >>> attention_module_time = GravitationalAttention(hidden_size, distance_type='time')
        >>> context_time, weights_time = attention_module_time(h_current, previous_hidden_states, 5)
        >>> context_time.shape
        torch.Size([4, 32])
        >>> weights_time.shape
        torch.Size([4, 10])
        """
        weight_func = self.calculate_weight
        distance_func = self.calculate_distance
        gravity_constant = self.gravity_constant
        distance_power = self.distance_power

        attention_weights = gravitational_attention_function(
            h_current,
            previous_hidden_states,
            weight_func,
            distance_func,
            gravity_constant,
            distance_power,
            timestep
        )

        context_vector = torch.bmm(attention_weights.transpose(1, 2), previous_hidden_states).squeeze(1)  # [batch_size, hidden_size]
        return context_vector, attention_weights.squeeze(-1)  # [batch_size, hidden_size], [batch_size, seq_len]

# LSTM cells with attention & Gravitational principle, class definition
class MomentumAttention(nn.Module):
    """
    Attention module calculating attention weights based on the principle of momentum.

    This module computes attention weights by considering the "momentum" of the
    current hidden state (relative to the previous one) and its "alignment"
    with previous hidden states, also factoring in the "distance" and "weight".

    Args:
        hidden_size (int): The number of features in the hidden state.
        weight_type (str, optional): Type of function to calculate the weight of a hidden state. Currently supports 'norm'. Defaults to 'norm'.
        distance_type (str, optional): Type of function to calculate the distance between hidden states. Options are 'euclidean' and 'time'. Defaults to 'time'.
        distance_lambda (float, optional): Coefficient controlling the influence of distance. Defaults to 0.1.
    """
    def __init__(self, hidden_size, weight_type='norm', distance_type='time', distance_lambda=0.1):
        """
        Initializes the MomentumAttention module.

        Args:
            hidden_size (int): The number of features in the hidden state.
            weight_type (str, optional): Type of function to calculate the weight ('norm'). Defaults to 'norm'.
            distance_type (str, optional): Type of function to calculate the distance ('euclidean', 'time'). Defaults to 'time'.
            distance_lambda (float, optional): Coefficient controlling the influence of distance. Defaults to 0.1.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_type = weight_type
        self.distance_type = distance_type
        self.distance_lambda = nn.Parameter(torch.tensor(distance_lambda))

    def calculate_weight(self, h):
        """
        Calculates the "weight" of a hidden state.

        Args:
            h (torch.Tensor): Hidden state tensor, shape: [..., hidden_size]

        Returns:
            torch.Tensor: Weight value, shape: [..., 1]

        Raises:
            ValueError: If an unknown weight type is specified.

        Examples:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> import torch.nn as nn
        
        >>> # Setting parameters
        >>> hidden_size = 8
        >>> batch_size = 2
        >>> h = torch.randn(batch_size, hidden_size)
        
        >>> # Initialize momentum attention with default weight type (norm)
        >>> attention = MomentumAttention(hidden_size)
        >>> attention.calculate_weight(h).shape
        torch.Size([2, 1])
        """
        if self.weight_type == 'norm':
            return torch.norm(h, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown weight type: {self.weight_type}")
        # ... other options for weight calculation

    def calculate_distance(self, h_current, previous_hidden_states):
        """
        Calculates the "distance" between the current and previous hidden states.

        Args:
            h_current (torch.Tensor): Current hidden state, shape: [batch_size, hidden_size]
            previous_hidden_states (torch.Tensor): Previous hidden states, shape: [batch_size, seq_len, hidden_size]

        Returns:
            torch.Tensor: Distance values, shape: [batch_size, seq_len]

        Raises:
            ValueError: If an unknown distance type is specified.

        Examples:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> import torch.nn as nn
        
        >>> # Setting parameters
        >>> hidden_size = 8
        >>> batch_size = 2
        >>> seq_len = 4
        >>> h_current = torch.randn(batch_size, hidden_size)
        >>> previous_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        >>> # Initialize momentum attention with default distance type (time)
        >>> attention = MomentumAttention(hidden_size)
        >>> attention.calculate_distance(h_current, previous_hidden_states).shape
        torch.Size([1, 4])
        >>> # Initialize momentum attention with distance type (euclidean)
        >>> attention = MomentumAttention(hidden_size, distance_type='euclidean')
        >>> attention.calculate_distance(h_current, previous_hidden_states).shape
        torch.Size([2, 4])
        """
        if self.distance_type == 'euclidean':
            return torch.cdist(h_current.unsqueeze(1), previous_hidden_states, p=2).squeeze(1).squeeze(-1) + 1e-8
        elif self.distance_type == 'time':
            batch_size, seq_len, _ = previous_hidden_states.size()
            return torch.arange(seq_len, 0, -1).float().to(h_current.device).unsqueeze(0) + 1e-8
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

    def forward(self, h_current, previous_hidden_states, h_prev):
        """
        Forward pass of the MomentumAttention module.

        Args:
            h_current (torch.Tensor): Current hidden state, shape: [batch_size, hidden_size]
            previous_hidden_states (torch.Tensor): Previous hidden states, shape: [batch_size, seq_len, hidden_size]
            h_prev (torch.Tensor): Previous hidden state (h_{t-2}), shape: [batch_size, hidden_size]

        Returns:
            tuple: A tuple containing:
                - context_vector (torch.Tensor): Context vector, shape: [batch_size, hidden_size]
                - attention_weights (torch.Tensor): Attention weights, shape: [batch_size, seq_len]

        Examples:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> import torch.nn as nn
        
        >>> # Setting parameters
        >>> hidden_size = 16
        >>> batch_size = 3
        >>> seq_len = 8
        >>> h_current = torch.randn(batch_size, hidden_size)
        >>> previous_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        >>> h_prev = torch.randn(batch_size, hidden_size)
        
        >>> # Initialize momentum attention with default parameters
        >>> attention_module = MomentumAttention(hidden_size)
        >>> context, weights = attention_module(h_current, previous_hidden_states, h_prev)
        
        >>> # Check the output shapes
        >>> context.shape
        torch.Size([3, 16])
        >>> weights.shape
        torch.Size([3, 8])
        
        >>> # Initialize momentum attention with specific parameters
        >>> attention_module_euclidean = MomentumAttention(hidden_size, distance_type='euclidean', distance_lambda=0.02)
        >>> context_euc, weights_euc = attention_module_euclidean(h_current, previous_hidden_states, h_prev)
        
        >>> # Check the output shapes
        >>> context_euc.shape
        torch.Size([3, 16])
        >>> weights_euc.shape
        torch.Size([3, 8])
        """
        weight_func = self.calculate_weight
        distance_func = self.calculate_distance
        distance_lambda = self.distance_lambda

        attention_weights = momentum_attention_function(
            h_current,
            previous_hidden_states,
            h_prev,
            weight_func,
            distance_func,
            distance_lambda
        )

        context_vector = torch.bmm(attention_weights.transpose(1, 2), previous_hidden_states).squeeze(1)
        return context_vector, attention_weights.squeeze(-1)
        


### Integration all as one class ###
# LSTM Cell with Attention, Class definition
class LSTMCellWithAttention(nn.Module):
    """A single LSTM cell with an attention mechanism over the previous hidden states.

    Args:
        input_size (int): The number of expected features in the input x.
        hidden_size (int): The number of features in the hidden state h.
        output_size (int): The number of features in the output.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """Initializes the LSTMCellWithAttention.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            output_size (int): The number of features in the output.
        """
        super(LSTMCellWithAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights for input-to-hidden transformations (gates)
        self.W_i = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))  # Input gate
        self.W_f = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))  # Forget gate
        self.W_c = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))  # Cell gate
        self.W_o = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))  # Output gate

        # Biases for gates
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        # Attention layer (optional: could use a simpler dot-product attention if preferred)
        self.attention = nn.Linear(hidden_size, hidden_size, bias=False)

        # Final fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        """Initializes the weights and biases of the LSTM cell."""
        # Initialize weights and biases using Xavier for weights and zeros for biases
        for weight in [self.W_i, self.W_f, self.W_c, self.W_o]:
            nn.init.xavier_uniform_(weight)
        for bias in [self.b_i, self.b_f, self.b_c, self.b_o]:
            nn.init.zeros_(bias)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, x, h_prev=None, c_prev=None):
        """
        Forward pass for the LSTM cell with attention over the sequence.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size), optional
            c_prev: Previous cell state (batch_size, hidden_size), optional

        Returns:
            output: Final output after fully connected layer (batch_size, output_size)
            h_t: Final hidden state (batch_size, hidden_size)
            c_t: Final cell state (batch_size, hidden_size)
        """
        batch_size, seq_length, _ = x.size()

        # Initialize hidden and cell states if not provided
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if c_prev is None:
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Store hidden states for attention
        hidden_seq = []
        previous_hidden_states = torch.zeros(batch_size, seq_length, self.hidden_size, device=x.device)

        h_t, c_t = h_prev, c_prev
        for t in range(seq_length):
            # Concatenate current input and previous hidden state
            combined = torch.cat((x[:, t, :], h_t), dim=1)

            # Compute gates
            i_t = torch.sigmoid(torch.mm(combined, self.W_i.t()) + self.b_i)
            f_t = torch.sigmoid(torch.mm(combined, self.W_f.t()) + self.b_f)
            c_tilde = torch.tanh(torch.mm(combined, self.W_c.t()) + self.b_c)
            o_t = torch.sigmoid(torch.mm(combined, self.W_o.t()) + self.b_o)

            # Update cell state
            c_t = f_t * c_t + i_t * c_tilde

            # Compute attention over previous hidden states up to this timestep
            relevant_hidden_states = previous_hidden_states[:, :t + 1, :]  # Shape: (batch_size, t + 1, hidden_size)
            if t > 0:  # Apply attention only if there are previous states
                # Compute attention scores
                attention_scores = torch.bmm(relevant_hidden_states, h_t.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, t + 1)
                attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(1)  # Shape: (batch_size, 1, t + 1)
                context_vector = torch.bmm(attention_weights, relevant_hidden_states).squeeze(1)  # Shape: (batch_size, hidden_size)
            else:
                context_vector = torch.zeros(batch_size, self.hidden_size, device=x.device)  # No context at t=0

            # Update hidden state with attention
            h_t = o_t * torch.tanh(c_t) + context_vector

            # Store the hidden state
            hidden_seq.append(h_t)
            previous_hidden_states[:, t, :] = h_t

        # Final output through fully connected layer
        output = self.fc(h_t)

        return output, h_t, c_t


# LSTM Cell with Attention and Entropic regularization, Class definition
class LSTMWithAttentionEntropyReg(nn.Module):
    """A single LSTM Cell with Attention Mechanism and Entropic Regularization over the previous hidden states.

    Args:
        input_size (int): The number of expected features in the input x.
        hidden_size (int): The number of features in the hidden state h.
        output_size (int): The number of features in the output.
        lambda_entropy (float): The weight for the entropic regularization term.
                                A higher value encourages more uniform attention weights.
    """    
    def __init__(self, input_size, hidden_size, output_size, lambda_entropy):
        """Initializes the LSTMWithAttentionEntropyReg.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            output_size (int): The number of features in the output.
            lambda_entropy (float): The weight for the entropic regularization term.
        """        
        super(LSTMWithAttentionEntropyReg, self).__init__()
        self.hidden_size = hidden_size
        self.lambda_entropy = lambda_entropy

        # LSTM weights and biases
        self.weights_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weights_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        # Attention weights (kept for compatibility, not used directly)
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initializes the weights and biases of the LSTM cell."""      
        nn.init.xavier_uniform_(self.weights_ih)
        nn.init.xavier_uniform_(self.weights_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)
        nn.init.xavier_uniform_(self.attention_weights)

    def lstm_cell_attention_entropy_reg(self, input_tensor, hidden_state, cell_state, previous_hidden_states, timestep):
        """Performs a single LSTM cell forward pass with attention and entropy regularization.

        Args:
            input_tensor (torch.Tensor): Input tensor for the current timestep (batch_size, input_size).
            hidden_state (torch.Tensor): Previous hidden state (batch_size, hidden_size).
            cell_state (torch.Tensor): Previous cell state (batch_size, hidden_size).
            previous_hidden_states (torch.Tensor): Tensor of all previous hidden states (batch_size, seq_length, hidden_size).
            timestep (int): The current timestep in the sequence (0-indexed).

        Returns:
            tuple: A tuple containing:
                - next_hidden_state (torch.Tensor): The next hidden state (batch_size, hidden_size).
                - next_cell_state (torch.Tensor): The next cell state (batch_size, hidden_size).
        """       
        # Linear transformation (fused)
        gates = torch.matmul(input_tensor, self.weights_ih.t()) + self.bias_ih + torch.matmul(hidden_state, self.weights_hh.t()) + self.bias_hh

        # Splitting into individual gates
        input_gate = torch.sigmoid(gates[:, 0:self.hidden_size])
        forget_gate = torch.sigmoid(gates[:, self.hidden_size:2 * self.hidden_size])
        output_gate = torch.sigmoid(gates[:, 2 * self.hidden_size:3 * self.hidden_size])
        cell_gate = torch.tanh(gates[:, 3 * self.hidden_size:4 * self.hidden_size])

        # Updating the cell state
        cell_state = forget_gate * cell_state + input_gate * cell_gate

        # Attention mechanism
        relevant_hidden_states = previous_hidden_states[:, :timestep + 1, :]  # Shape: (batch_size, timestep + 1, hidden_size)
        if timestep > 0:  # Only apply attention if there are previous states
            # Compute attention scores over the sequence
            attention_scores = torch.bmm(relevant_hidden_states, hidden_state.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, timestep + 1)
            attention_weights_normalized = F.softmax(attention_scores, dim=1)  # Shape: (batch_size, timestep + 1)
            context_vector = torch.bmm(attention_weights_normalized.unsqueeze(1), relevant_hidden_states).squeeze(1)  # Shape: (batch_size, hidden_size)

            # Calculate entropy of attention weights
            entropy = -torch.sum(attention_weights_normalized * torch.log(attention_weights_normalized + 1e-8), dim=1).mean()
        else:
            context_vector = torch.zeros_like(hidden_state, device=hidden_state.device)  # No attention at t=0
            entropy = torch.tensor(0.0, device=hidden_state.device)

        # Update hidden state with context vector and entropy regularization
        hidden_state = output_gate * torch.tanh(cell_state) + context_vector - self.lambda_entropy * entropy

        return hidden_state, cell_state

    def forward(self, x, h_prev, c_prev):
        """Performs the forward pass over the entire input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
            h_prev (torch.Tensor): Initial hidden state (batch_size, hidden_size).
            c_prev (torch.Tensor): Initial cell state (batch_size, hidden_size).

        Returns:
            tuple: A tuple containing:
                - out (torch.Tensor): Output of the fully connected layer at the last timestep (batch_size, output_size).
                - h_t (torch.Tensor): Final hidden state (batch_size, hidden_size).
                - c_t (torch.Tensor): Final cell state (batch_size, hidden_size).
        """        
        batch_size, seq_length, _ = x.size()
        hidden_seq = []
        previous_hidden_states = torch.zeros(batch_size, seq_length, self.hidden_size, device=x.device)

        h_t, c_t = h_prev, c_prev
        for t in range(seq_length):
            h_t, c_t = self.lstm_cell_attention_entropy_reg(
                x[:, t, :], h_t, c_t, previous_hidden_states, t
            )
            hidden_seq.append(h_t)
            previous_hidden_states[:, t, :] = h_t

        out = self.fc(hidden_seq[-1])
        return out, h_t, c_t


# LSTM Cell with Attention and Time Decay, Class definition
class LSTMWithAttentionTimeDecay(nn.Module):
    """LSTM Cell with Attention Mechanism and Time Decay.

    Args:
        input_size (int): The number of expected features in the input x.
        hidden_size (int): The number of features in the hidden state h.
        output_size (int): The number of features in the output.
        time_decay (float): A scalar that controls the rate of time decay.
                            A positive value will exponentially decrease the
                            importance of earlier hidden states.
    """    
    def __init__(self, input_size, hidden_size, output_size, time_decay):
        """Initializes the LSTMWithAttentionTimeDecay.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            output_size (int): The number of features in the output.
            time_decay (float): A scalar that controls the rate of time decay.
        """        
        super(LSTMWithAttentionTimeDecay, self).__init__()
        self.hidden_size = hidden_size
        self.time_decay = time_decay

        # LSTM weights and biases
        self.weights_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weights_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        # Attention weights (kept for compatibility, not used directly)
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initializes the weights and biases of the LSTM cell."""        
        nn.init.xavier_uniform_(self.weights_ih)
        nn.init.xavier_uniform_(self.weights_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)
        nn.init.xavier_uniform_(self.attention_weights)

    def lstm_cell_attention_time_decay(self, input_tensor, hidden_state, cell_state, previous_hidden_states, timestep):
        """Performs a single LSTM cell forward pass with attention and time decay.

        Args:
            input_tensor (torch.Tensor): Input tensor for the current timestep (batch_size, input_size).
            hidden_state (torch.Tensor): Previous hidden state (batch_size, hidden_size).
            cell_state (torch.Tensor): Previous cell state (batch_size, hidden_size).
            previous_hidden_states (torch.Tensor): Tensor of all previous hidden states (batch_size, seq_length, hidden_size).
            timestep (int): The current timestep in the sequence (0-indexed).

        Returns:
            tuple: A tuple containing:
                - next_hidden_state (torch.Tensor): The next hidden state (batch_size, hidden_size).
                - next_cell_state (torch.Tensor): The next cell state (batch_size, hidden_size).
        """        
        # Linear transformation (fused)
        gates = torch.matmul(input_tensor, self.weights_ih.t()) + self.bias_ih + torch.matmul(hidden_state, self.weights_hh.t()) + self.bias_hh

        # Splitting into individual gates
        input_gate = torch.sigmoid(gates[:, 0:self.hidden_size])
        forget_gate = torch.sigmoid(gates[:, self.hidden_size:2 * self.hidden_size])
        output_gate = torch.sigmoid(gates[:, 2 * self.hidden_size:3 * self.hidden_size])
        cell_gate = torch.tanh(gates[:, 3 * self.hidden_size:4 * self.hidden_size])

        # Updating the cell state
        cell_state = forget_gate * cell_state + input_gate * cell_gate

        # Attention mechanism
        relevant_hidden_states = previous_hidden_states[:, :timestep + 1, :]  # Shape: (batch_size, timestep + 1, hidden_size)
        if timestep > 0:  # Only apply attention if there are previous states
            # Compute attention scores
            attention_scores = torch.bmm(relevant_hidden_states, hidden_state.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, timestep + 1)
            attention_weights_unnormalized = torch.exp(attention_scores)  # Using exponential for non-negative weights

            # Apply time decay to attention weights
            time_weights = torch.exp(-self.time_decay * torch.arange(timestep + 1).float().to(input_tensor.device))
            time_weights = time_weights.unsqueeze(0)  # Add batch dimension
            attention_weights_with_decay = attention_weights_unnormalized * time_weights

            # Normalize attention weights with decay
            attention_weights_normalized = attention_weights_with_decay / torch.sum(attention_weights_with_decay, dim=1, keepdim=True)

            # Compute context vector
            context_vector = torch.bmm(attention_weights_normalized.unsqueeze(1), relevant_hidden_states).squeeze(1)  # Shape: (batch_size, hidden_size)
        else:
            context_vector = torch.zeros_like(hidden_state, device=hidden_state.device)  # No attention at t=0

        # Update hidden state with context vector
        hidden_state = output_gate * torch.tanh(cell_state) + context_vector

        return hidden_state, cell_state

    def forward(self, x, h_prev, c_prev):
        """Performs the forward pass over the entire input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
            h_prev (torch.Tensor): Initial hidden state (batch_size, hidden_size).
            c_prev (torch.Tensor): Initial cell state (batch_size, hidden_size).

        Returns:
            tuple: A tuple containing:
                - out (torch.Tensor): Output of the fully connected layer at the last timestep (batch_size, output_size).
                - h_t (torch.Tensor): Final hidden state (batch_size, hidden_size).
                - c_t (torch.Tensor): Final cell state (batch_size, hidden_size).
        """        
        batch_size, seq_length, _ = x.size()
        hidden_seq = []
        previous_hidden_states = torch.zeros(batch_size, seq_length, self.hidden_size, device=x.device)

        h_t, c_t = h_prev, c_prev
        for t in range(seq_length):
            h_t, c_t = self.lstm_cell_attention_time_decay(
                x[:, t, :], h_t, c_t, previous_hidden_states, t
            )
            hidden_seq.append(h_t)
            previous_hidden_states[:, t, :] = h_t

        out = self.fc(hidden_seq[-1])
        return out, h_t, c_t

# LSTM Cell with Attention and Mixed Mechanism, Class definitions
class LSTMWithAttentionMixed(nn.Module):
    """LSTM Cell with a Mixed Attention Mechanism (Time Decay, Entropy Regularization, and Randomness).

    This class implements a single LSTM cell with an attention mechanism
    over the previous hidden states in the sequence. It combines several
    techniques to influence the attention weights:
    - Time Decay: Reduces the impact of older hidden states.
    - Entropy Regularization: Encourages the attention distribution to be more uniform.
    - Randomness: Introduces stochasticity in the attention weights with a given probability.

    Args:
        input_size (int): The number of expected features in the input x.
        hidden_size (int): The number of features in the hidden state h.
        output_size (int): The number of features in the output.
        lambda_entropy (float): The weight for the entropic regularization term.
                                A higher value encourages more uniform attention weights.
        time_decay (float): A scalar that controls the rate of time decay.
                            A positive value will exponentially decrease the
                            importance of earlier hidden states.
        random_prob (float): The probability (between 0 and 1) of applying
                             random masking to the attention weights at each timestep.
    """ 
    def __init__(self, input_size, hidden_size, output_size, lambda_entropy, time_decay, random_prob):
        """Initializes the LSTMWithAttentionMixed.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            output_size (int): The number of features in the output.
            lambda_entropy (float): The weight for the entropic regularization term.
            time_decay (float): A scalar that controls the rate of time decay.
            random_prob (float): The probability of applying random masking to attention weights.
        """        
        super(LSTMWithAttentionMixed, self).__init__()
        self.hidden_size = hidden_size
        self.lambda_entropy = lambda_entropy
        self.time_decay = time_decay
        self.random_prob = random_prob

        # LSTM weights and biases
        self.weights_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weights_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        # Attention weights (kept for compatibility, not used directly)
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initializes the weights and biases of the LSTM cell.""" 
        nn.init.xavier_uniform_(self.weights_ih)
        nn.init.xavier_uniform_(self.weights_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)
        nn.init.xavier_uniform_(self.attention_weights)

    def lstm_cell_attention_mixed(self, input_tensor, hidden_state, cell_state, previous_hidden_states, timestep):
        """Performs a single LSTM cell forward pass with mixed attention mechanisms.

        Args:
            input_tensor (torch.Tensor): Input tensor for the current timestep (batch_size, input_size).
            hidden_state (torch.Tensor): Previous hidden state (batch_size, hidden_size).
            cell_state (torch.Tensor): Previous cell state (batch_size, hidden_size).
            previous_hidden_states (torch.Tensor): Tensor of all previous hidden states (batch_size, seq_length, hidden_size).
            timestep (int): The current timestep in the sequence (0-indexed).

        Returns:
            tuple: A tuple containing:
                - next_hidden_state (torch.Tensor): The next hidden state (batch_size, hidden_size).
                - next_cell_state (torch.Tensor): The next cell state (batch_size, hidden_size).
        """        
        # Linear transformation (fused)
        gates = torch.matmul(input_tensor, self.weights_ih.t()) + self.bias_ih + torch.matmul(hidden_state, self.weights_hh.t()) + self.bias_hh

        # Splitting into individual gates
        input_gate = torch.sigmoid(gates[:, 0:self.hidden_size])
        forget_gate = torch.sigmoid(gates[:, self.hidden_size:2 * self.hidden_size])
        output_gate = torch.sigmoid(gates[:, 2 * self.hidden_size:3 * self.hidden_size])
        cell_gate = torch.tanh(gates[:, 3 * self.hidden_size:4 * self.hidden_size])

        # Updating the cell state
        cell_state = forget_gate * cell_state + input_gate * cell_gate

        # Attention mechanism
        relevant_hidden_states = previous_hidden_states[:, :timestep + 1, :]  # Shape: (batch_size, timestep + 1, hidden_size)
        if timestep > 0:  # Only apply attention if there are previous states
            # Compute attention scores
            attention_scores = torch.bmm(relevant_hidden_states, hidden_state.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, timestep + 1)
            attention_weights_unnormalized = torch.exp(attention_scores)  # Using exponential for non-negative weights

            # Apply time decay to attention weights
            time_weights = torch.exp(-self.time_decay * torch.arange(timestep + 1).float().to(input_tensor.device))
            time_weights = time_weights.unsqueeze(0)  # Add batch dimension
            attention_weights_with_decay = attention_weights_unnormalized * time_weights

            # Normalize attention weights with decay
            attention_weights_normalized_base = attention_weights_with_decay / torch.sum(attention_weights_with_decay, dim=1, keepdim=True)

            # Apply randomness
            if random.random() < self.random_prob:
                random_mask = torch.rand(attention_weights_normalized_base.shape, device=input_tensor.device) < 0.5
                attention_weights_normalized = attention_weights_normalized_base.masked_fill(random_mask, 0)
                attention_weights_normalized = attention_weights_normalized / (torch.sum(attention_weights_normalized, dim=1, keepdim=True) + 1e-8)  # Re-normalize
            else:
                attention_weights_normalized = attention_weights_normalized_base

            # Compute context vector
            context_vector = torch.bmm(attention_weights_normalized.unsqueeze(1), relevant_hidden_states).squeeze(1)  # Shape: (batch_size, hidden_size)

            # Calculate entropy of attention weights
            entropy = -torch.sum(attention_weights_normalized * torch.log(attention_weights_normalized + 1e-8), dim=1).mean()
        else:
            context_vector = torch.zeros_like(hidden_state, device=hidden_state.device)  # No attention at t=0
            entropy = torch.tensor(0.0, device=hidden_state.device)

        # Update hidden state with context vector and entropy regularization
        hidden_state = output_gate * torch.tanh(cell_state) + context_vector - self.lambda_entropy * entropy

        return hidden_state, cell_state

    def forward(self, x, h_prev, c_prev):
        """Performs the forward pass over the entire input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
            h_prev (torch.Tensor): Initial hidden state (batch_size, hidden_size).
            c_prev (torch.Tensor): Initial cell state (batch_size, hidden_size).

        Returns:
            tuple: A tuple containing:
                - out (torch.Tensor): Output of the fully connected layer at the last timestep (batch_size, output_size).
                - h_t (torch.Tensor): Final hidden state (batch_size, hidden_size).
                - c_t (torch.Tensor): Final cell state (batch_size, hidden_size).
        """        
        batch_size, seq_length, _ = x.size()
        hidden_seq = []
        previous_hidden_states = torch.zeros(batch_size, seq_length, self.hidden_size, device=x.device)

        h_t, c_t = h_prev, c_prev
        for t in range(seq_length):
            h_t, c_t = self.lstm_cell_attention_mixed(
                x[:, t, :], h_t, c_t, previous_hidden_states, t
            )
            hidden_seq.append(h_t)
            previous_hidden_states[:, t, :] = h_t

        out = self.fc(hidden_seq[-1])
        return out, h_t, c_t


# LSTM Cell with Attention and Potential Difference Principle, Class definition
class LSTMWithAttentionPD(nn.Module):
    """LSTM Cell with Attention based on Potential Difference Principle.

    This module implements an LSTM cell where the attention mechanism is
    inspired by the potential difference principle. The attention weights
    are determined by the "potential difference" between the current hidden
    state and previous hidden states, modulated by "resistances" in time
    and semantic space.

    Args:
        input_size (int): The number of expected features in the input x.
        hidden_size (int): The number of features in the hidden state h.
        output_size (int): The number of features in the output.
        potential_type (str, optional): The method to calculate the potential
                                         of a hidden state. Options are 'norm' (L2 norm),
                                         'linear' (linear projection), or 'tanh' (tanh
                                         activation after linear projection).
                                         Defaults to 'norm'.
        time_resistance_lambda (float, optional): Scaling factor for the time
                                                  resistance. Higher values make
                                                  older states have exponentially
                                                  higher resistance. Defaults to 0.1.
        semantic_resistance_lambda (float, optional): Scaling factor for the
                                                      semantic resistance based on
                                                      cosine dissimilarity. Higher
                                                      values penalize semantically
                                                      different past states more.
                                                      Defaults to 1.0.
        """    
    def __init__(self, input_size, hidden_size, output_size, potential_type='norm', time_resistance_lambda=0.1, semantic_resistance_lambda=1.0):
        """Initializes the LSTMWithAttentionPD module.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            output_size (int): The number of features in the output.
            potential_type (str, optional): Method to calculate potential. Defaults to 'norm'.
            time_resistance_lambda (float, optional): Scaling for time resistance. Defaults to 0.1.
            semantic_resistance_lambda (float, optional): Scaling for semantic resistance. Defaults to 1.0.
        """        
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.potential_type = potential_type

        # LSTM parameters
        self.lstm_cell = nn.LSTMCell(input_size + hidden_size, hidden_size)  # Input + Context

        # Attention parameters
        self.time_resistance_lambda = nn.Parameter(torch.tensor(float(time_resistance_lambda)))
        self.semantic_resistance_lambda = nn.Parameter(torch.tensor(float(semantic_resistance_lambda)))
        self.wp = nn.Linear(hidden_size, 1)  # For linear/tanh potential types

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the LSTM and attention layers."""        
        # Initialize LSTM weights (handled by nn.LSTMCell)
        for name, param in self.lstm_cell.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        # Initialize attention weights
        nn.init.xavier_uniform_(self.wp.weight)
        nn.init.zeros_(self.wp.bias)

    def potential(self, h):
        """
        Calculate the potential of a hidden state.

        Args:
            h (torch.Tensor): Hidden state tensor, shape: [..., hidden_size]

        Returns:
            torch.Tensor: Potential value, shape: [..., 1]
        """
        if self.potential_type == 'norm':
            return torch.norm(h, dim=-1, keepdim=True)
        elif self.potential_type == 'linear':
            return self.wp(h)
        elif self.potential_type == 'tanh':
            return torch.tanh(self.wp(h))
        else:
            raise ValueError(f"Unknown potential type: {self.potential_type}")

    def time_resistance(self, time_diff):
        """
        Calculate time resistance based on time difference.

        Args:
            time_diff (torch.Tensor): Time difference tensor, shape: [1, seq_len, 1]

        Returns:
            torch.Tensor: Time resistance, shape: [1, seq_len, 1]
        """
        return torch.exp(self.time_resistance_lambda * time_diff)

    def semantic_resistance(self, h_current, h_past):
        """
        Calculate semantic resistance between current and past hidden states.

        Args:
            h_current (torch.Tensor): Current hidden state, shape: [batch_size, 1, hidden_size]
            h_past (torch.Tensor): Past hidden states, shape: [batch_size, seq_len, hidden_size]

        Returns:
            torch.Tensor: Semantic resistance, shape: [batch_size, seq_len, 1]
        """
        cos_sim = F.cosine_similarity(h_current, h_past, dim=-1, eps=1e-8)
        return torch.exp(self.semantic_resistance_lambda * (1 - cos_sim.unsqueeze(-1)))

    def forward(self, x, h_prev=None, c_prev=None):
        """
        Forward pass for the LSTM with potential difference-based attention.

        Args:
            x (torch.Tensor): Input tensor, shape: [batch_size, seq_length, input_size]
            h_prev (torch.Tensor, optional): Initial hidden state, shape: [batch_size, hidden_size]
            c_prev (torch.Tensor, optional): Initial cell state, shape: [batch_size, hidden_size]

        Returns:
            tuple: (output, h_t, c_t, attention_history)
                - output: Final output, shape: [batch_size, output_size]
                - h_t: Final hidden state, shape: [batch_size, hidden_size]
                - c_t: Final cell state, shape: [batch_size, hidden_size]
                - attention_history: Attention weights, shape: [batch_size, seq_length, seq_length]
        """
        batch_size, seq_length, _ = x.size()

        # Initialize hidden and cell states if not provided
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if c_prev is None:
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)

        hidden_seq = []
        attention_weights_seq = []
        previous_hidden_states = torch.zeros(batch_size, seq_length, self.hidden_size, device=x.device)

        h_t, c_t = h_prev, c_prev
        for t in range(seq_length):
            # Attention computation
            potential_current = self.potential(h_t)  # [batch_size, 1]
            potential_past = self.potential(previous_hidden_states)  # [batch_size, seq_len, 1]

            # Time resistance
            time_diff = torch.arange(seq_length - 1, -1, -1).float().to(x.device).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            time_resistance = self.time_resistance(time_diff)  # [1, seq_len, 1]

            # Semantic resistance
            semantic_resistance = self.semantic_resistance(h_t.unsqueeze(1), previous_hidden_states)  # [batch_size, seq_len, 1]

            # Total resistance
            resistance = time_resistance + semantic_resistance + 1e-8  # [batch_size, seq_len, 1]

            # Potential difference
            potential_difference = torch.abs(potential_current.unsqueeze(1) - potential_past)  # [batch_size, seq_len, 1]

            # Information flow
            flow = potential_difference / resistance  # [batch_size, seq_len, 1]

            # Mask future timesteps
            mask = torch.ones(batch_size, seq_length, 1, device=x.device)
            mask[:, t + 1 :, :] = -float('inf')  # Mask future steps
            flow = flow + mask  # Apply mask

            # Normalize to attention weights
            attention_weights = F.softmax(flow, dim=1).squeeze(-1)  # [batch_size, seq_len]

            # Compute context vector
            context_vector = torch.bmm(attention_weights.unsqueeze(1), previous_hidden_states).squeeze(1)  # [batch_size, hidden_size]

            # LSTM step
            lstm_input = torch.cat((x[:, t, :], context_vector), dim=1)  # [batch_size, input_size + hidden_size]
            h_t, c_t = self.lstm_cell(lstm_input, (h_t, c_t))

            # Store results
            hidden_seq.append(h_t)
            attention_weights_seq.append(attention_weights)
            previous_hidden_states[:, t, :] = h_t

        # Final output
        out = self.fc(hidden_seq[-1])
        attention_history = torch.stack(attention_weights_seq, dim=1)  # [batch_size, seq_length, seq_length]

        return out, h_t, c_t, attention_history


# LSTM Cell with Attention and Gravitational Principles, Class definition
class LSTMWithGravitationalAttention(nn.Module):
    """LSTM Cell with Attention based on Gravitational Principles.

    This module implements an LSTM cell where the attention mechanism is
    inspired by the law of universal gravitation. The attention weights
    are determined by the 'mass' (weight) of the current and previous
    hidden states and the 'distance' between them.

    Args:
        input_size (int): The number of expected features in the input x.
        hidden_size (int): The number of features in the hidden state h.
        output_size (int): The number of features in the output.
        gravity_constant (float, optional): The gravitational constant scaling
                                           the attraction force. Defaults to 1.0.
        distance_power (float, optional): The exponent for the distance in the
                                         gravitational force calculation.
                                         Defaults to 2.0 (inverse square law).
        weight_type (str, optional): The method to calculate the 'mass' (weight)
                                     of a hidden state. Options are 'norm' (L2 norm)
                                     or 'entropy' (entropy of the hidden state
                                     interpreted as a probability distribution).
                                     Defaults to 'norm'.
        distance_type (str, optional): The method to calculate the 'distance'
                                       between hidden states. Options are 'euclidean'
                                       (Euclidean distance) or 'time' (temporal
                                       distance, i.e., the time difference).
                                       Defaults to 'euclidean'.
    """    
    def __init__(self, input_size, hidden_size, output_size, gravity_constant=1.0, distance_power=2.0, weight_type='norm', distance_type='euclidean'):
        """Initializes the LSTMWithGravitationalAttention module.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            output_size (int): The number of features in the output.
            gravity_constant (float, optional): Gravitational constant. Defaults to 1.0.
            distance_power (float, optional): Exponent for distance. Defaults to 2.0.
            weight_type (str, optional): Method to calculate weight. Defaults to 'norm'.
            distance_type (str, optional): Method to calculate distance. Defaults to 'euclidean'.
        """        
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_type = weight_type
        self.distance_type = distance_type

        # LSTM parameters
        self.lstm_cell = nn.LSTMCell(input_size + hidden_size, hidden_size)  # Input + Context

        # Attention parameters
        self.gravity_constant = nn.Parameter(torch.tensor(float(gravity_constant)))
        self.distance_power = nn.Parameter(torch.tensor(float(distance_power)))

        # Weight function parameters (for non-norm weight types)
        self.weight_linear = nn.Linear(hidden_size, 1) if weight_type != 'norm' else None

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the LSTM and attention layers."""        
        # Initialize LSTM weights
        for name, param in self.lstm_cell.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        # Initialize weight function weights if applicable
        if self.weight_linear is not None:
            nn.init.xavier_uniform_(self.weight_linear.weight)
            nn.init.zeros_(self.weight_linear.bias)

    def calculate_weight(self, h):
        """
        Calculate the 'weight' of a hidden state.

        Args:
            h (torch.Tensor): Hidden state tensor, shape: [..., hidden_size]

        Returns:
            torch.Tensor: Weight value, shape: [..., 1]
        """
        if self.weight_type == 'norm':
            return torch.norm(h, dim=-1, keepdim=True)
        elif self.weight_type == 'entropy':
            p = F.softmax(h, dim=-1)
            return -(p * torch.log(p + 1e-8)).sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown weight type: {self.weight_type}")

    def calculate_distance(self, h_current, previous_hidden_states):
        """
        Calculate the 'distance' between current and previous hidden states.

        Args:
            h_current (torch.Tensor): Current hidden state, shape: [batch_size, hidden_size]
            previous_hidden_states (torch.Tensor): Previous hidden states, shape: [batch_size, seq_len, hidden_size]

        Returns:
            torch.Tensor: Distance tensor, shape: [batch_size, seq_len]
        """
        if self.distance_type == 'euclidean':
            # Compute pairwise Euclidean distances
            diff = h_current.unsqueeze(1) - previous_hidden_states  # [batch_size, seq_len, hidden_size]
            return torch.norm(diff, dim=-1) + 1e-8  # [batch_size, seq_len]
        elif self.distance_type == 'time':
            _, seq_len, _ = previous_hidden_states.size()
            return torch.arange(seq_len, 0, -1).float().to(h_current.device).unsqueeze(0) + 1e-8  # [1, seq_len]
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

    def forward(self, x, h_prev=None, c_prev=None):
        """
        Forward pass for the LSTM with gravitational attention.

        Args:
            x (torch.Tensor): Input tensor, shape: [batch_size, seq_length, input_size]
            h_prev (torch.Tensor, optional): Initial hidden state, shape: [batch_size, hidden_size]
            c_prev (torch.Tensor, optional): Initial cell state, shape: [batch_size, hidden_size]

        Returns:
            tuple: (output, h_t, c_t, attention_history)
                - output: Final output, shape: [batch_size, output_size]
                - h_t: Final hidden state, shape: [batch_size, hidden_size]
                - c_t: Final cell state, shape: [batch_size, hidden_size]
                - attention_history: Attention weights, shape: [batch_size, seq_length, seq_length]
        """
        batch_size, seq_length, _ = x.size()

        # Initialize hidden and cell states if not provided
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if c_prev is None:
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)

        hidden_seq = []
        attention_weights_seq = []
        previous_hidden_states = torch.zeros(batch_size, seq_length, self.hidden_size, device=x.device)

        h_t, c_t = h_prev, c_prev
        for t in range(seq_length):
            # Compute weights
            weight_current = self.calculate_weight(h_t)  # [batch_size, 1]
            weight_past = self.calculate_weight(previous_hidden_states)  # [batch_size, seq_len, 1]

            # Compute distances
            distance = self.calculate_distance(h_t, previous_hidden_states)  # [batch_size, seq_len]

            # Ensure distance has correct shape
            if distance.ndim == 2:
                distance = distance.unsqueeze(-1)  # [batch_size, seq_len, 1]
            else:
                raise ValueError(f"Unexpected distance tensor shape: {distance.shape}")

            # Calculate gravitational force
            gravity_force = self.gravity_constant * (weight_current.unsqueeze(1) * weight_past) / (distance.pow(self.distance_power) + 1e-8)  # [batch_size, seq_len, 1]

            # Mask future timesteps
            mask = torch.ones(batch_size, seq_length, 1, device=x.device)
            mask[:, t + 1 :, :] = -float('inf')  # Mask future steps
            gravity_force = gravity_force + mask  # Apply mask

            # Normalize to attention weights
            attention_weights = F.softmax(gravity_force, dim=1).squeeze(-1)  # [batch_size, seq_len]

            # Compute context vector
            context_vector = torch.bmm(attention_weights.unsqueeze(1), previous_hidden_states).squeeze(1)  # [batch_size, hidden_size]

            # LSTM step
            lstm_input = torch.cat((x[:, t, :], context_vector), dim=1)  # [batch_size, input_size + hidden_size]
            h_t, c_t = self.lstm_cell(lstm_input, (h_t, c_t))

            # Store results
            hidden_seq.append(h_t)
            attention_weights_seq.append(attention_weights)
            previous_hidden_states[:, t, :] = h_t

        # Final output
        out = self.fc(hidden_seq[-1])
        attention_history = torch.stack(attention_weights_seq, dim=1)  # [batch_size, seq_length, seq_length]

        return out, h_t, c_t, attention_history


class LSTMMomentumAttention(nn.Module):
    """Attention mechanism inspired by the principle of momentum.

    This module calculates attention weights based on the momentum of the
    current hidden state relative to the previous one, the 'weight' of the
    hidden states, and the 'distance' between them.

    Args:
        hidden_size (int): The number of features in the hidden state h.
        weight_type (str, optional): The method to calculate the 'weight' of a
                                     hidden state. Options are 'norm' (L2 norm).
                                     Defaults to 'norm'.
        distance_type (str, optional): The method to calculate the 'distance'
                                       between hidden states. Options are 'euclidean'
                                       (Euclidean distance) or 'time' (temporal
                                       distance, i.e., the inverse of the
                                       position in the sequence). Defaults to 'time'.
        distance_lambda (float, optional): Scaling factor for the distance in
                                         the attention score calculation.
                                         Defaults to 0.1.
    """    
    def __init__(self, hidden_size, weight_type='norm', distance_type='time', distance_lambda=0.1):
        """Initializes the MomentumAttention module.

        Args:
            hidden_size (int): The number of features in the hidden state h.
            weight_type (str, optional): Method to calculate weight. Defaults to 'norm'.
            distance_type (str, optional): Method to calculate distance. Defaults to 'time'.
            distance_lambda (float, optional): Scaling for distance in score. Defaults to 0.1.
        """        
        super(LSTMMomentumAttention, self).__init__()
        self.hidden_size = hidden_size
        self.weight_type = weight_type
        self.distance_type = distance_type
        self.distance_lambda = nn.Parameter(torch.tensor(distance_lambda))

    def calculate_weight(self, h):
        """Calculate the 'weight' of a hidden state.

        Args:
            h (torch.Tensor): Hidden state tensor, shape: [..., hidden_size]

        Raises:
            ValueError: If an unknown weight type is specified.

        Returns:
            torch.Tensor: Weight value, shape: [..., 1]
        """        
        if self.weight_type == 'norm':
            return torch.norm(h, dim=-1, keepdim=True)
        # Add other weight calculation options if needed
        else:
            raise ValueError(f"Unknown weight type: {self.weight_type}")

    def calculate_distance(self, h_current, previous_hidden_states):
        """Calculate the 'distance' between current and previous hidden states.

        Args:
            h_current (torch.Tensor): Current hidden state, shape: [batch_size, hidden_size]
            previous_hidden_states (torch.Tensor): Previous hidden states, shape: [batch_size, seq_len, hidden_size]

        Raises:
            ValueError: If an unknown distance type is specified.

        Returns:
            torch.Tensor: Distance tensor, shape: [batch_size, seq_len]
        """        
        if self.distance_type == 'euclidean':
            return torch.cdist(h_current.unsqueeze(0), previous_hidden_states, p=2).squeeze(0) + 1e-8
        elif self.distance_type == 'time':
            batch_size, seq_len, _ = previous_hidden_states.size()
            return torch.arange(seq_len, 0, -1).float().to(h_current.device).unsqueeze(0) + 1e-8
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

    def momentum_attention_function(self, h_current, previous_hidden_states, h_prev):
        """
        Calculation of attention weights based on the principle of momentum.
        """
        batch_size, seq_len, hidden_size = previous_hidden_states.size()

        weight_current = self.calculate_weight(h_current)  # [batch_size, 1]
        weight_past = self.calculate_weight(previous_hidden_states)  # [batch_size, seq_len, 1]

        # Calculation of "velocity"
        velocity = (h_current - h_prev.detach()).unsqueeze(1)  # [batch_size, 1, hidden_size]

        # Calculation of "momentum" (direction)
        momentum = weight_current.unsqueeze(1) * velocity  # [batch_size, 1, hidden_size]

        # Calculation of "alignment" with previous states
        alignment = torch.bmm(momentum, previous_hidden_states.transpose(1, 2)).squeeze(1)  # [batch_size, seq_len]

        # Calculation of distance
        distance = self.calculate_distance(h_current, previous_hidden_states).squeeze(-1)  # [batch_size, seq_len]

        # Calculation of score
        score = alignment + weight_past.squeeze(-1) - self.distance_lambda * distance

        # Normalization to attention weights
        attention_weights = F.softmax(score, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]

        return attention_weights

    def forward(self, h_current, previous_hidden_states, h_prev):
        """Forward pass for the Momentum Attention mechanism.

        Args:
            h_current (torch.Tensor): The current hidden state (batch_size, hidden_size).
            previous_hidden_states (torch.Tensor): All previous hidden states
                (batch_size, seq_len, hidden_size).
            h_prev (torch.Tensor): The hidden state from the previous timestep
                (batch_size, hidden_size).

        Returns:
            tuple: A tuple containing:
                - context_vector (torch.Tensor): The context vector (batch_size, hidden_size).
                - attention_weights (torch.Tensor): The attention weights
                  (batch_size, seq_len).
        """        
        attention_weights = self.momentum_attention_function(h_current, previous_hidden_states, h_prev)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), previous_hidden_states).squeeze(1)
        return context_vector, attention_weights.squeeze(-1)


if __name__ == "__main__":
    import doctest
    doctest.testmod()