from typing import Optional
from einops import rearrange
import jax
import jax.numpy as jnp
from flax import linen as nn

class CausalAttention(nn.Module):
    dim: int
    num_heads: int
    ff_dim: int
    rope_frequency: float = 10000
    pad_mask: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self, x, pad_mask=None):
        x_norm = nn.LayerNorm()(x)

        # Q, K, V projection
        Q = nn.Dense(self.dim, name="q_proj")(x_norm)
        K = nn.Dense(self.dim, name="k_proj")(x_norm)
        V = nn.Dense(self.dim, name="v_proj")(x_norm)

        # Split into heads
        head_dim = self.dim // self.num_heads

        Qh = rearrange(
            Q,
            (
                "batch sequence (num_heads head_dim) -> "
                "batch num_heads sequence head_dim"
            ),
            num_heads=self.num_heads,
            head_dim=head_dim,
        )

        Kh = rearrange(
            K,
            (
                "batch sequence (num_heads head_dim) -> "
                "batch num_heads sequence head_dim"
            ),
            num_heads=self.num_heads,
            head_dim=head_dim,
        )

        Vh = rearrange(
            V,
            (
                "batch sequence (num_heads head_dim) -> "
                "batch num_heads sequence head_dim"
            ),
            num_heads=self.num_heads,
            head_dim=head_dim,
        )

        # Apply RoPE
        pos_indices = jnp.arange(Qh.shape[2])
        freq_indices = jnp.arange(head_dim // 2)
        inv_freq = 1 / jnp.pow(self.rope_frequency, 2 * freq_indices /
                               head_dim)

        theta = jnp.outer(pos_indices, inv_freq)
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)

        # Broadcast sin and cos
        sin_theta = sin_theta[None, None, :, :]
        cos_theta = cos_theta[None, None, :, :]

        # Split Q and K
        Q_even = Qh[..., 0::2]
        Q_odd = Qh[..., 1::2]
        K_even = Kh[..., 0::2]
        K_odd = Kh[..., 1::2]

        # Rotate Q and K
        Q_even_rotated = Q_even * cos_theta - Q_odd * sin_theta
        Q_odd_rotated = Q_even * sin_theta + Q_odd * cos_theta
        K_even_rotated = K_even * cos_theta - K_odd * sin_theta
        K_odd_rotated = K_even * sin_theta + K_odd * cos_theta

        # Interleave odd and even elements
        Q_interleaved = jnp.stack((Q_even_rotated, Q_odd_rotated),
                                  axis=-1).reshape(Qh.shape)
        K_interleaved = jnp.stack((K_even_rotated, K_odd_rotated),
                                  axis=-1).reshape(Kh.shape)
        # Calculate attention
        K_transpose = rearrange(
            K_interleaved,
            (
                "batch num_heads sequence head_dim -> "
                "batch num_heads head_dim sequence"
            ),
        )
        scores = jnp.matmul(Q_interleaved, K_transpose)
        scores = scores/jnp.sqrt(head_dim)

        causal_mask = jnp.tri(Q_interleaved.shape[-2], dtype=bool)
        if (pad_mask is not None):
            mask = jnp.logical_and(causal_mask[None, None, :, :],
                                   pad_mask[:, None, None, :])
        else:
            mask = causal_mask[None, None, :, :]

        mask = jnp.where(mask, 0.0, -jnp.inf)
        scores = jax.nn.softmax(scores + mask, axis=-1)
        attention = jnp.matmul(scores, Vh)

        # merge heads
        attention = rearrange(
            attention,
            (
                "batch num_heads sequence head_dim -> "
                "batch sequence (num_heads head_dim)"
            ),
        )
        out = nn.Dense(self.dim, name="out_proj")(attention)
        x = x + out

        # MLP Block
        x_norm = nn.LayerNorm()(x)
        mlp_out = nn.Dense(self.ff_dim, name="expansion_proj")(x_norm)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(self.dim, name="contraction_proj")(mlp_out)
        x = x + mlp_out

        return x


class CausalTransformer(nn.Module):

    vocab_size: int
    dim: int
    num_layers: int
    num_heads: int
    ff_dim: int
    rope_frequency: float = 10000
    PAD_ID: Optional[int] = None

    def setup(self):
        if (self.dim % self.num_heads != 0):
            raise ValueError("Number of heads should divide dimension of the\
            model!")
        if ((self.dim // self.num_heads) % 2 == 1):
            raise ValueError("Dimension of heads should be even for RoPE!")

        self.token_embedding_table = nn.Embed(self.vocab_size, self.dim)
        self.transformer_layers = [CausalAttention(dim=self.dim,
                                                   num_heads=self.num_heads,
                                                   ff_dim=self.ff_dim,
                                                   rope_frequency=self.rope_frequency)
                                   for _ in range(self.num_layers)]
        self.final_ln = nn.LayerNorm()

    def __call__(self, x):
        if(self.PAD_ID is not None):
            pad_mask = (x != self.PAD_ID)
        else:
            pad_mask = None

        x = self.token_embedding_table(x)
        
        for layer in self.transformer_layers:
            x = layer(x, pad_mask=pad_mask)

        x = self.final_ln(x)
        embedding_matrix = self.token_embedding_table.embedding
        logits = x @ embedding_matrix.T
        return logits
