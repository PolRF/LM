import math
from typing import List, Literal
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np


@dataclass
class ModelConfig:
    vocab_size: int = (
        50304  # --> This is the size of the tiktoken tokenizer for gpt2 model (50257 tokens but 50304 is the nearest multiple of 64)
    )
    block_size: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_kv_heads: int = 4
    n_layer: int = 12
    dropout: float = 0.1
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pos_emb: Literal["rope", "alibi"] = "rope"
    theta: float = 10000.0
    max_seq_len: int = 1024
    max_batch_size: int = 64

    # Mixture of Experts
    num_experts: int = 1
    num_experts_per_token: int | None = None


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit.
    """

    # Implementation following the paper https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
                )
            )
        )


def _rope_frequency(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
) -> torch.Tensor:
    """
    Frequency tensor for the rotary position embedding.
    θi = 10000^(−2i/d)
    """
    assert (
        head_dim % 2 == 0
    ), "The dimension of the frequency tensor must be even"
    # -2i in vector form is the same of vector starting from 0, until dim with 2 steps
    # [: (head_dim // 2)] is used subtract the last element if the head_dim is odd
    numerator = torch.arange(0, head_dim, 2)[: (head_dim // 2)].float()
    frequencies = 1.0 / (theta ** (numerator / head_dim)).to(device)

    # Position
    p = torch.arange(seq_len, device=device)

    # Multiply each theta by the position (outer product)
    # This will create a matrix (seq_len, head_dim/2)
    frequencies = torch.outer(p, frequencies).float()

    # define the same matrix with all 1
    matrix = torch.ones_like(frequencies)
    # We then have to construct the polars.
    f = torch.polar(matrix, frequencies)
    return f


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the rotary position embedding to the input tensor.
    """
    # # Reshape the input tensor to a complex tensor
    # # (B, Seq_Len, Head_Dim) -> (B, Seq_Len, Head_Dim/2, 2)
    # # x.shape[:-1] is the batch and the sequence length (get the first two dimensions of the tensor)
    # # -1 is used as a placeholder that will automatically be calculated based on the size of the tensor and the remaining dimensions
    # # 2 is the last dimension of the tensor. We will get a vector with size 2 for each element in the tensor
    # # view_as_complex() will convert the tensor to a complex tensor --> The 2 elements specified before,
    # # will be used as the real and imaginary part of the complex number
    # q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    # k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # # Multiply the input tensor by the frequency tensor to apply the rotary position embedding
    # # Final shape will be (B, Seq_Len, H, Head_Dim/2)
    # q_rotated = q_complex * freqs_complex
    # k_rotated = k_complex * freqs_complex

    # # Convert again to real tensor
    # q_out = torch.view_as_real(q_rotated)
    # k_out = torch.view_as_real(k_rotated)
    # # Reshape the tensor to the original shape
    # # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    # q_out = q_out.reshape(*q.shape)
    # k_out = k_out.reshape(*k.shape)
    # # Convert the tensor to the original type and move it to the original device
    # return q_out.type_as(q).to(device), k_out.type_as(k).to(device)

    # Extract dimensions
    batch_size, seq_len, n_heads, head_dim = q.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    cos, sin = freqs_cos, freqs_sin
    # Reshape q and k to separate the last dimension into two halves
    q_reshaped = q.view(batch_size, seq_len, n_heads, 2, -1)
    k_reshaped = k.view(batch_size, seq_len, n_heads, 2, -1)

    # Extract even and odd indices
    q_even, q_odd = q_reshaped[:, :, :, 0], q_reshaped[:, :, :, 1]
    k_even, k_odd = k_reshaped[:, :, :, 0], k_reshaped[:, :, :, 1]

    # Compute cosine and sine

    # Apply rotation
    q_out_even = q_even * cos - q_odd * sin
    q_out_odd = q_odd * cos + q_even * sin
    k_out_even = k_even * cos - k_odd * sin
    k_out_odd = k_odd * cos + k_even * sin

    # Concatenate even and odd parts
    q_out = torch.stack([q_out_even, q_out_odd], dim=3).view(*q.shape)
    k_out = torch.stack([k_out_even, k_out_odd], dim=3).view(*k.shape)

    # Convert the tensor to the original type and move it to the original device
    return q_out.type_as(q).to(device), k_out.type_as(k).to(device)


class DecoderMultiHeadAttention(nn.Module):
    """
    Also known as Casual Self Attention.
    This is the attention head that will be used in the decoder.
    The attention head will take the input and compute the attention weights
    based on the query, key, and value.
    The future tokens are masked so that the attention head only looks at
    the previous tokens (this is what makes it a decoder attention head).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        # Instead of defining the linear layers for the query, key, and value separately
        # we define a single linear layer that will output the query, key, and value.
        # This is done to improve the performance of the model.
        self.att = nn.Linear(config.n_embd, config.n_embd * 3, bias=False)
        self.n_head = config.n_head
        self.flash = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )
        if not self.flash:
            # Register buffer as it is not a parameter. The ones with the tril (lower triangular) matrix with 1s
            # is used to mask the upper triangular part of the matrix. We just want the attention to look at the
            # previous tokens and not the future tokens. This will aggregate the means of the previous tokens.
            self.register_buffer(
                "tril",
                torch.tril(torch.ones(config.block_size, config.block_size)),
            )

        self.linear_projection = nn.Linear(
            config.n_embd, config.n_embd, bias=True
        )
        self.projection_dropout = nn.Dropout(config.dropout)
        # Add dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor):
        # batch is the size of the batch
        # tokens is the number of tokens in the sequence
        # head_size is the size of the hidden layer in the attention head
        B, T, C = x.shape  # batch, tokens, channels
        # Split the output of the linear layer into query, key, and value
        # The output of the linear layer will be of size (B, T, C)
        k, q, v = self.att(x).chunk(3, dim=-1)
        # Modify each vector to fit the shape including the head size
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # batch, head size, tokens, channels // head size
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # batch, head size, tokens, channels // head size
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # batch, head size, tokens, channels // head size

        # Apply the rotary position embedding
        q, k = apply_rope(q, k, rope_freqs, str(x.device))

        if self.flash:
            # Don't apply custom mask as the param is_causal already apply the mask
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            # batch, head_size, tokens, channels // head_size
        else:

            # Compute the attention weights
            # k.transpose(2,3) --> batch, head_size, tokens, channels // head_size
            # q @ k.transpose(2,3) will compute the dot product of the query and the key.-> batch, head_size, tokens, tokens
            # Here is where the magic happens. We compute the attention weights by taking the dot product of the query
            # and the key. We then divide by the square root of the head size. This is the scaled dot product attention.
            wei = (q @ k.transpose(2, 3)) * (
                k.shape[-1] ** -0.5
            )  # batch, head_size, tokens, tokens
            # we have to mask the upper triangular part of the matrix
            # we do this by adding a large negative number to the places we want to mask
            # and then take the softmax of the matrix.
            # This is done so that the softmax of the upper triangular part of the matrix is 0 when softmaxed
            # and the other parts are not affected.
            # self.tril[:T, :T] == 0 will return a matrix of the same size as wei with 1s (True) in the upper triangular part
            # masked_fill() will replace the 1s (True) with -inf.
            wei = wei.masked_fill(
                self.tril[:T, :T] == 0, float("-inf")
            )  # batch, head_size, tokens, tokens

            # apply softmax and dropout
            # apply the softmax to the last dimension of the matrix
            wei = F.softmax(wei, dim=-1)  # batch, head_size, tokens, tokens
            wei = self.attn_dropout(wei)

            # This means that values will pass through the attention head and the output will be the weighted sum of the values
            # based on the attention weights.
            output = wei @ v  # batch, head_size, tokens, channels // head_size
        # First we need to reshape again the output of the attention head
        # Reshape the output of the attention head to the original shape
        # (B, T, H, C//H) -> (B, T, C)
        # Use contiguous() to make sure that the tensor is stored in a contiguous block of memory
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        # Apply the linear projection to combine the output of the attention heads
        output = self.projection_dropout(self.linear_projection(output))
        return output


class DecoderGroupedQueryHeadAttentionRope(nn.Module):
    """
    Decoder only attention head with grouped query heads instead of multi-head attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_heads
        self.head_dim = config.n_embd // config.n_head
        self.q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.kv = nn.Linear(
            config.n_embd, self.head_dim * config.n_kv_heads * 2, bias=False
        )
        self.q_kv_proportion = config.n_head // config.n_kv_heads
        self.flash = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )
        assert self.flash
        self.linear_projection = nn.Linear(
            config.n_embd, config.n_embd, bias=True
        )
        self.projection_dropout = nn.Dropout(config.dropout)
        # Add dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # Cache
        self.cache_k = torch.zeros(
            config.max_batch_size,
            config.max_seq_len,
            config.n_kv_heads,
            self.head_dim,
            device=config.device,
        )
        self.cache_v = torch.zeros(
            config.max_batch_size,
            config.max_seq_len,
            config.n_kv_heads,
            self.head_dim,
            device=config.device,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        start_position: int = 0,
    ):
        B, T, C = x.shape
        q = self.q(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k, v = self.kv(x).chunk(2, dim=-1)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        with torch.autocast(enabled=False, device_type=str("cuda")):
            q, k = apply_rope(q, k, rope_cos, rope_sin, str(x.device))

        # Cache the keys and values:
        # self.cache_k[:B, start_position : start_position + T] = k
        # self.cache_v[:B, start_position : start_position + T] = v

        # k = self.cache_k[:B, : start_position + T]
        # v = self.cache_v[:B, : start_position + T]

        # Repeat the keys and values to match query heads
        k = k.repeat_interleave(self.q_kv_proportion, dim=1)
        v = v.repeat_interleave(self.q_kv_proportion, dim=1)

        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.projection_dropout(self.linear_projection(output))
        return output


class DecoderGroupedQueryHeadAttentionAlibi(nn.Module):
    """
    Decoder only attention head with grouped query heads instead of multi-head attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_heads
        self.head_dim = config.n_embd // config.n_head
        self.q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.kv = nn.Linear(
            config.n_embd, self.head_dim * config.n_kv_heads * 2, bias=False
        )
        self.q_kv_proportion = config.n_head // config.n_kv_heads
        self.flash = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )
        assert self.flash
        self.linear_projection = nn.Linear(
            config.n_embd, config.n_embd, bias=True
        )
        self.projection_dropout = nn.Dropout(config.dropout)
        # Add dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # Alibi
        alibi = self.gen_alibi_mask(
            config.max_seq_len, self.n_head, device=config.device
        )
        causal_mask = torch.tril(
            torch.ones(
                self.max_seq_len, self.max_seq_len, device=config.device
            )
        )
        causal_mask = causal_mask.view(1, self.max_seq_len, self.max_seq_len)
        alibi_bias = alibi.masked_fill(causal_mask == 0, float("-inf"))
        self.register_buffer(
            "alibi_mask",
            torch.tril(alibi_bias),
        )

    def get_slopes_power_of_2(self, n_head):
        """
        Generate slopes based on the number of heads.
        Following the paper:
        8 heads  --> 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256
        16 heads --> 1/(2^0.5), 1/2, 1/(2^1.5), 1/4, 1/(2^2.5), 1/8, 1/(2^3.5)... 1/256
        """
        assert n_head % 2 == 0
        start = 2 ** (-(2 ** -(math.log2(n_head) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n_head)]

    def gen_alibi_mask(self, seq_len, n_head, device):
        """
        1. Generate mask:
        [0, -1, -2, -3, -4, -5, -6]
        [-1, 0, -1, -2, -3, -4, -5]
        [-2, -1, 0, -1, -2, -3, -4]
        [-3, -2, -1, 0, -1, -2, -3]
        [-4, -3, -2, -1, 0, -1, -2]
        [-5, -4, -3, -2, -1, 0, -1]
        [-6, -5, -4, -3, -2, -1, 0]

        2. Apply the slopes to the mask.
        """
        # Generate the initial ALiBi mask with shape (seq_len, seq_len)
        alibi_mask = torch.arange(
            seq_len, device=device, dtype=torch.float
        ).unsqueeze(0) - torch.arange(
            seq_len, device=device, dtype=torch.float
        ).unsqueeze(
            1
        )

        # Get slopes and scale the mask for each head
        slopes = torch.tensor(
            self.get_slopes_power_of_2(n_head),
            dtype=torch.float,
            device=device,
        )
        alibi_mask = alibi_mask.unsqueeze(0) * slopes.view(n_head, 1, 1)

        return alibi_mask

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        q = self.q(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k, v = self.kv(x).chunk(2, dim=-1)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Repeat the keys and values to match query heads
        k = k.repeat(1, self.q_kv_proportion, 1, 1)
        v = v.repeat(1, self.q_kv_proportion, 1, 1)

        # Apply the alibi mask with the scaling factor to match the scaling
        # of scaled dot product attention
        alibi_bias = self.alibi_mask[:, :T, :T] * (self.head_dim**-0.5)

        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            alibi_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.projection_dropout(self.linear_projection(output))
        return output


class MixtureOfExpertsLayer(nn.Module):
    """
    Mixture of experts layer.
    Experts are just FNN layers.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        assert config.num_experts > 1
        assert config.num_experts_per_token is not None

        self.experts = nn.ModuleList(
            [FFN(config) for _ in range(config.num_experts)]
        )
        # Gate is the router that will select the experts for each token
        self.gate = nn.Linear(config.n_embd, config.num_experts, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = inputs.shape
        inputs = inputs.view(-1, self.config.n_embd)
        routes = self.gate(inputs)
        # Select the top k experts for each token
        # where k is the num of experts per token we want to use
        assert self.config.num_experts_per_token is not None

        weights = F.softmax(routes, dim=1, dtype=torch.float).to(inputs.dtype)
        weights, selected_experts = torch.topk(
            weights, self.config.num_experts_per_token
        )
        weights /= weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        weights = weights.to(inputs.dtype)
        # Apply softmax to the weights
        results = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=weights.dtype,
            device=weights.device,
        )

        # Create a mask for the selected experts
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.config.num_experts
        ).permute(2, 1, 0)
        for expert_idx in range(self.config.num_experts):
            # Get the expert layer
            expert_layer = self.experts[expert_idx]
            # Get the indices of the tokens that will be routed to the expert
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `weights` on the corresponding tokens (top-1 and top-2)
            current_state = inputs[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * weights[top_x, idx, None]
            )

            results.index_add_(
                0, top_x, current_hidden_states.to(inputs.dtype)
            )
        results = results.reshape(batch_size, sequence_length, hidden_dim)
        return results


class FFN(nn.Module):
    """
    Feed forward network.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=True),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=True),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """
    The attention block is the core of the transformer. It contains the multihead attention
    and the feed forward network.

    We also will aggregate x and the output of the attention block.
    This will help "skip" the network and allow the gradients to flow through the network.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_rope = config.pos_emb == "rope"
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        if config.n_kv_heads == config.n_head:
            self.attn = DecoderMultiHeadAttention(config)
        elif self.use_rope:
            self.attn = DecoderGroupedQueryHeadAttentionRope(config)
        elif config.pos_emb == "alibi":
            self.attn = DecoderGroupedQueryHeadAttentionAlibi(config)

        if config.num_experts > 1:
            self.ffn = MixtureOfExpertsLayer(config)
        else:
            self.ffn = FFN(config)

        # RoPE
        # We need to initialize the frequency tensor for the rotary position embedding
        if self.use_rope:
            rope_frequencies = _rope_frequency(
                config.n_embd // config.n_head,
                config.block_size,
                theta=config.theta,
                device=str(config.device),
            )
            self.rope_cos = rope_frequencies.cos()
            self.rope_sin = rope_frequencies.sin()

    def forward(self, x: torch.Tensor, start_pos: int):
        B, Seq_len, C = x.shape
        # Take into account that we apply the normalization before the attention block
        # This is a modification from original paper Attention is All You Need
        # (a better implementation)
        if self.use_rope:
            x = x + self.attn(
                self.ln1(x), self.rope_cos[:Seq_len], self.rope_sin[:Seq_len], start_pos
            )
        else:
            x = x + self.attn(self.ln1(x))

        x = x + self.ffn(self.ln2(x))
        return x


class GPTLM(nn.Module):
    """
    This is the final models.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.block_size = config.block_size
        self.config = config
        # The token embedding is in charge of converting the token (int of the dict) to a vector
        # with semantic value.
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # The positional embedding is in charge of inferring the relative position of the
        # tokens in the sequence. This is important because the transformer is not recurrent
        # and does not have the notion of order in the sequence.
        # DEPRECATED: self.positional_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [AttentionBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer normalization
        # This is the linear layer that will convert the output of the transformer to the output vocabulary
        # later, we will convert the int to the respective text token.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # The weight for the token embeddings are the same as the weights for the last linear layer
        # that maps the output of the transformer to the output vocabulary.
        self.token_embedding.weight = self.lm_head.weight

        # This is totally copypaste from the Karpathy's implementation.
        # We need to initialize the weights of the Linear layers and the Embedding layers
        # with a normal distribution with mean 0 and std 0.02 so that it's easier to train the model.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, start_pos: int = 0):
        B, Seq_len = idx.shape
        # Get the device of the input
        # device = idx.device
        assert (
            Seq_len <= self.block_size
        ), f"Cannot forward sequence of length {Seq_len}, block size is only {self.block_size}"

        # idx and targets are both (B,T) tensor of integers
        x = self.token_embedding(idx)  # (B,Seq_len, C)
        # pos_emb = self.positional_embedding(torch.arange(T, device=device)) # (T,C)
        # DEPRECATED (we use RoPE Now): Add the positional embedding to the token embedding
        # x = x + pos_emb # (B,Seq_len,C)
        for block in self.blocks:
            x = block(x, start_pos)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        # This is for training the model. We will compute the loss
        # by comparing the logits with the targets.
        if targets is None:
            loss = None
        else:
            B, Seq_len, C = logits.shape
            # Flatten the logits and the targets
            logits = logits.view(B * Seq_len, C)
            targets = targets.view(B * Seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        This function will generate new tokens based on the input tokens.
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


################### HUGGING FACE ADAPTER ####################
from dataclasses import fields
import torch
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from typing import Literal


class GPTConfig(PretrainedConfig):
    model_type = "gpt"
    tokenizer_class = "gpt-2"

    def __init__(
        self,
        vocab_size: int = (
            50304  # --> This is the size of the tiktoken tokenizer for gpt2 model (50257 tokens but 50304 is the nearest multiple of 64)
        ),
        block_size: int = 1024,
        n_embd: int = 768,
        n_head: int = 12,
        n_kv_heads: int = 4,
        n_layer: int = 12,
        dropout: float = 0.1,
        device: Literal["cuda", "cpu"] = "cpu",
        pos_emb: Literal["rope", "alibi"] = "rope",
        theta: int = 10_000,
        max_seq_len: int = 1024,
        max_batch_size: int = 32,
        # Mixture of Experts
        num_experts: int = 1,
        num_experts_per_token: int | None = None,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_heads = n_kv_heads
        self.n_layer = n_layer
        self.dropout = dropout
        self.device = device
        self.pos_emb = pos_emb
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token


def from_gptconfig_to_modelconfig(config: GPTConfig) -> ModelConfig:
    model_config_fields = {f.name for f in fields(ModelConfig)}
    model_config_args = {
        k: v for k, v in config.__dict__.items() if k in model_config_fields
    }
    assert len(model_config_args) == len(model_config_fields)
    model_config_args["device"] = torch.device(config.device)
    return ModelConfig(**model_config_args)


class GPTModel(PreTrainedModel):
    config_class = GPTConfig

    def __init__(self, model_config: GPTConfig):
        super().__init__(model_config)
        # Get the field names of ModelConfig
        adapted_config = from_gptconfig_to_modelconfig(model_config)
        self.model = GPTLM(adapted_config)

    def tie_weights(self):
        self.model.lm_head.weight = self.model.token_embedding.weight

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
