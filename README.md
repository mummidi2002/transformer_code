#Mummidi Sathwika
#700776842

1. Scaled Dot-Product Attention (NumPy)
Overview

This part implements the Scaled Dot-Product Attention mechanism described in the Transformer architecture.
Given Query (Q), Key (K), and Value (V) matrices, the function computes:

Attention(Q,K,V)=softmax(dk​
​QKT​)V
Features

Uses NumPy for matrix operations

Includes a numerically stable softmax implementation

Returns both attention weights and the context vector

Matches the equations described in Attention is All You Need

Files

scaled_dot_product_attention.py — contains the implementation.

How to Run
python scaled_dot_product_attention.py

Expected Output

The script prints:

Attention weights (softmax-normalized scores)

Context vector (weighted sum of V)

2. Simple Transformer Encoder Block (PyTorch)
Overview

This part implements a simplified Transformer Encoder Block using PyTorch.
The architecture includes:

Multi-Head Self-Attention

Add & Layer Normalization

Feed-Forward Network (Linear → ReLU → Linear)

Second Add & Layer Normalization

This matches the encoder block described in the Transformer model PDF.

Model Hyperparameters

d_model = 128

num_heads = 8

Feed-Forward hidden size: 512

Uses batch_first=True for readability

Files

simple_transformer_encoder.py — contains the encoder block implementation and a shape test.

How to Run
python simple_transformer_encoder.py

Expected Output

You should see:

Input shape:  torch.Size([32, 10, 128])
Output shape: torch.Size([32, 10, 128])


This verifies:

Residual connections work correctly

LayerNorm preserves shape

The encoder block outputs a transformed representation with the same d_model dimension

Environment Requirements
For NumPy Code
numpy

For PyTorch Code
torch


Install using:

pip install numpy torch
