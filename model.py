import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np





@dataclass
class ModelConfig:
    vocab_size: int = 50304 # --> This is the size of the tiktoken tokenizer for gpt2 model (50257 tokens but 50304 is the nearest multiple of 64)
    block_size: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.2

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit.
    """
    # Implementation following the paper https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))

class DecoderAttentionHead(nn.Module):
    """
    This is the attention head that will be used in the decoder.
    The attention head will take the input and compute the attention weights
    based on the query, key, and value.
    The future tokens are masked so that the attention head only looks at 
    the previous tokens (this is what makes it a decoder attention head).
    """
    def __init__(self, config:ModelConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.k = nn.Linear(config.n_embd, head_size, bias=False)
        self.v = nn.Linear(config.n_embd, head_size, bias=False)
        self.q = nn.Linear(config.n_embd, head_size, bias=False)
        # Register buffer as it is not a parameter. The ones with the tril (lower triangular) matrix with 1s
        # is used to mask the upper triangular part of the matrix. We just want the attention to look at the
        # previous tokens and not the future tokens. This will aggregate the means of the previous tokens.
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))

        # Add dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
    
    def forward(self,x):
        # batch is the size of the batch
        # tokens is the number of tokens in the sequence
        # head_size is the size of the hidden layer in the attention head
        B,T,C = x.shape # batch, tokens, channels

        k = self.k(x) # batch, tokens, head_size
        v = self.v(x) # batch, tokens, head_size
        q = self.q(x) # batch, tokens, head_size
        # Compute the attention weights
        # k.transpose(1,2) --> batch, head_size, tokens
        # q @ k.transpose(1,2) will compute the dot product of the query and the key.-> batch, tokens, tokens
        # Here is where the magic happens. We compute the attention weights by taking the dot product of the query
        # and the key. We then divide by the square root of the head size. This is the scaled dot product attention. 
        wei = q @ k.transpose(1,2) * (k.shape[-1] ** -0.5) # batch, tokens, tokens

        # we have to mask the upper triangular part of the matrix
        # we do this by adding a large negative number to the places we want to mask
        # and then take the softmax of the matrix. 
        # This is done so that the softmax of the upper triangular part of the matrix is 0 when softmaxed
        # and the other parts are not affected.
        # self.tril[:T, :T] == 0 will return a matrix of the same size as wei with 1s (True) in the upper triangular part
        # masked_fill() will replace the 1s (True) with -inf.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)

        # apply softmax and dropout
        # apply the softmax to the last dimension of the matrix
        wei = F.softmax(wei, dim=-1) # batch, tokens, tokens
        wei = self.attn_dropout(wei)

        # This means that values will pass through the attention head and the output will be the weighted sum of the values
        # based on the attention weights. 
        output = wei @ v # batch, tokens, head_size

        return output


class MultiHeadAttention(nn.Module):
    """
    Multihead attention is the concatenation of multiple attention heads.
    The attention heads are run in parallel and the outputs are concatenated.
    """
    def __init__(self, config:ModelConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        # Define a list of modules. Each module is an attention head. Create n_head attention heads
        self.attn_heads = nn.ModuleList([DecoderAttentionHead(config) for _ in range(config.n_head)])
        # Define a linear layer to combine the output of the attention heads
        self.linear_projection = nn.Linear(config.n_head * head_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        # Run each attention head in parallel
        # torch.cat() will concatenate the output of each attention head along the last dimension
        out = torch.cat([h(x) for h in self.attn_heads], dim=-1)
        out = self.dropout(self.linear_projection(out))
        return out
    

class FFN(nn.Module):
    """
    Feed forward network.
    """
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.GELU(),
            nn.Linear(4*config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
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
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ffn = FFN(config)
    
    def forward(self,x):
        # Take into account that we apply the normalization before the attention block
        # This is a modification from original paper Attention is All You Need
        # (a better implementation)
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    

class GPTLM(nn.Module):
    """
    This is the final models.
    """
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.block_size = config.block_size
        self.config = config
        # The token embedding is in charge of converting the token (int of the dict) to a vector
        # with semantic value.
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # The positional embedding is in charge of inferring the relative position of the 
        # tokens in the sequence. This is important because the transformer is not recurrent
        # and does not have the notion of order in the sequence.
        self.positional_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[AttentionBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd) # final layer normalization
        # This is the linear layer that will convert the output of the transformer to the output vocabulary
        # later, we will convert the int to the respective text token.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

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
    

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Get the device of the input
        device = idx.device
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding(idx) # (B,T,C)
        pos_emb = self.positional_embedding(torch.arange(T, device=device)) # (T,C)
        # Add the positional embedding to the token embedding
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        
        # This is for training the model. We will compute the loss
        # by comparing the logits with the targets.
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # Flatten the logits and the targets
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        This function will generate new tokens based on the input tokens.
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
