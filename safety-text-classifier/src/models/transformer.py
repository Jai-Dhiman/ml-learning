"""
Safety Text Classifier Transformer Model

JAX/Flax implementation of a transformer-based safety text classifier
for the Constitutional AI research project.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import Dense, Dropout, LayerNorm, Embed
from typing import Callable, Optional, Tuple, Any
import numpy as np
from functools import partial


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    num_heads: int
    head_dim: int
    dropout_rate: float = 0.1

    def setup(self):
        self.dense_q = Dense(self.num_heads * self.head_dim, use_bias=False)
        self.dense_k = Dense(self.num_heads * self.head_dim, use_bias=False)
        self.dense_v = Dense(self.num_heads * self.head_dim, use_bias=False)
        self.dense_output = Dense(self.num_heads * self.head_dim)
        self.dropout = Dropout(self.dropout_rate)

    def __call__(self, x, mask=None, training=True):
        batch_size, seq_len, embed_dim = x.shape

        # Compute queries, keys, values
        q = self.dense_q(x)
        k = self.dense_k(x)
        v = self.dense_v(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Compute attention scores
        attention_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))
        attention_scores = attention_scores / jnp.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            mask = jnp.expand_dims(mask, axis=1)  # (batch_size, 1, seq_len)
            mask = jnp.expand_dims(mask, axis=1)  # (batch_size, 1, 1, seq_len)

            # Apply mask (large negative value for masked positions)
            attention_scores = jnp.where(mask, attention_scores, -1e9)

        # Apply softmax
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, deterministic=not training)

        # Apply attention to values
        attention_output = jnp.matmul(attention_weights, v)

        # Transpose back and reshape
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))
        attention_output = attention_output.reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )

        # Final linear projection
        output = self.dense_output(attention_output)

        return output, attention_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    hidden_dim: int
    output_dim: int
    dropout_rate: float = 0.1

    def setup(self):
        self.dense1 = Dense(self.hidden_dim)
        self.dense2 = Dense(self.output_dim)
        self.dropout = Dropout(self.dropout_rate)

    def __call__(self, x, training=True):
        x = self.dense1(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, deterministic=not training)
        x = self.dense2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""

    num_heads: int
    head_dim: int
    feedforward_dim: int
    dropout_rate: float = 0.1

    def setup(self):
        embed_dim = self.num_heads * self.head_dim

        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
        )
        self.feed_forward = FeedForward(
            hidden_dim=self.feedforward_dim,
            output_dim=embed_dim,
            dropout_rate=self.dropout_rate,
        )
        self.layer_norm1 = LayerNorm()
        self.layer_norm2 = LayerNorm()
        self.dropout = Dropout(self.dropout_rate)

    def __call__(self, x, mask=None, training=True):
        # Multi-head attention with residual connection
        attn_output, attn_weights = self.attention(x, mask=mask, training=training)
        attn_output = self.dropout(attn_output, deterministic=not training)
        x = self.layer_norm1(x + attn_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x, training=training)
        ff_output = self.dropout(ff_output, deterministic=not training)
        x = self.layer_norm2(x + ff_output)

        return x, attn_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    max_length: int
    embed_dim: int

    def setup(self):
        # Create positional encoding matrix
        position = jnp.arange(self.max_length)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.embed_dim, 2) * -(jnp.log(10000.0) / self.embed_dim)
        )

        pe = jnp.zeros((self.max_length, self.embed_dim))
        pe = pe.at[:, ::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        self.pe = pe

    def __call__(self, x):
        seq_len = x.shape[1]
        # Handle sequences longer than max_length by creating pe on-the-fly
        if seq_len > self.max_length:
            position = jnp.arange(seq_len)[:, None]
            div_term = jnp.exp(
                jnp.arange(0, self.embed_dim, 2) * -(jnp.log(10000.0) / self.embed_dim)
            )
            pe = jnp.zeros((seq_len, self.embed_dim))
            pe = pe.at[:, ::2].set(jnp.sin(position * div_term))
            pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
            return x + pe
        else:
            return x + self.pe[:seq_len]


class SafetyTransformer(nn.Module):
    """
    Transformer-based safety text classifier.

    Attributes:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of token embeddings
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        feedforward_dim: Dimension of feed-forward layers
        max_sequence_length: Maximum input sequence length
        num_classes: Number of safety categories to classify
        dropout_rate: Dropout rate for regularization
    """

    vocab_size: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    feedforward_dim: int
    max_sequence_length: int
    num_classes: int
    dropout_rate: float = 0.1

    def setup(self):
        self.head_dim = self.embedding_dim // self.num_heads
        assert (
            self.embedding_dim % self.num_heads == 0
        ), "embedding_dim must be divisible by num_heads"

        # Embedding layers
        self.token_embedding = Embed(
            num_embeddings=self.vocab_size, features=self.embedding_dim
        )
        self.positional_encoding = PositionalEncoding(
            max_length=self.max_sequence_length, embed_dim=self.embedding_dim
        )

        # Transformer layers
        self.transformer_blocks = [
            TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                feedforward_dim=self.feedforward_dim,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers)
        ]

        # Classification head
        self.layer_norm = LayerNorm()
        self.dropout = Dropout(self.dropout_rate)
        self.classifier = Dense(self.num_classes)

    def create_attention_mask(self, input_ids):
        """Create attention mask from input_ids (assuming 0 is padding token)."""
        return input_ids != 0

    def __call__(self, input_ids, training=True):
        """
        Forward pass of the safety transformer.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            training: Whether in training mode

        Returns:
            logits: Safety classification logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len = input_ids.shape

        # Create attention mask
        attention_mask = self.create_attention_mask(input_ids)

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x, deterministic=not training)

        # Pass through transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(
                x, mask=attention_mask, training=training
            )
            attention_weights.append(attn_weights)

        # Apply final layer norm
        x = self.layer_norm(x)

        # Global average pooling over sequence dimension
        # Use attention mask to avoid pooling over padding tokens
        mask_expanded = jnp.expand_dims(attention_mask, axis=-1)
        x_masked = x * mask_expanded
        seq_lengths = jnp.sum(attention_mask, axis=1, keepdims=True)
        pooled = jnp.sum(x_masked, axis=1) / jnp.maximum(seq_lengths, 1)

        # Classification
        pooled = self.dropout(pooled, deterministic=not training)
        logits = self.classifier(pooled)

        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "hidden_states": x,
        }


def create_model(config: dict) -> SafetyTransformer:
    """
    Create a SafetyTransformer model from configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized SafetyTransformer model
    """
    model_config = config["model"]

    return SafetyTransformer(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        feedforward_dim=model_config["feedforward_dim"],
        max_sequence_length=model_config["max_sequence_length"],
        num_classes=model_config["num_classes"],
        dropout_rate=model_config["dropout_rate"],
    )


def initialize_model(
    model: SafetyTransformer,
    rng_key: jax.random.PRNGKey,
    input_shape: Optional[Tuple[int, int]] = None,
) -> Any:
    """
    Initialize model parameters.

    Args:
        model: SafetyTransformer model
        rng_key: Random key for initialization
        input_shape: Shape of input (batch_size, seq_len). If None, uses model's max_sequence_length

    Returns:
        Initialized model parameters
    """
    if input_shape is None:
        input_shape = (1, model.max_sequence_length)
    
    dummy_input = jnp.ones(input_shape, dtype=jnp.int32)
    params = model.init({'params': rng_key}, dummy_input)
    return params


if __name__ == "__main__":
    # Test model creation and initialization
    import yaml

    # Load config
    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create model
    model = create_model(config)

    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    params = initialize_model(model, rng)

    # Test forward pass
    dummy_input = jnp.ones((2, 512), dtype=jnp.int32)
    output = model.apply(params, dummy_input, training=False)

    print(f"Model created successfully!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Number of attention layers: {len(output['attention_weights'])}")

    # Count parameters
    def count_params(params):
        return sum(x.size for x in jax.tree_util.tree_leaves(params))

    param_count = count_params(params)
    print(f"Total parameters: {param_count:,}")
