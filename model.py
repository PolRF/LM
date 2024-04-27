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
    device:torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit.
    """
    # Implementation following the paper https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))


    

def _rope_frequency(head_dim:int,seq_len:int, device:str, theta:float = 10000.0)-> torch.Tensor:
    """
    Frequency tensor for the rotary position embedding.
    θi = 10000^(−2i/d)
    """
    assert head_dim % 2 == 0, "The dimension of the frequency tensor must be even"
    # -2i in vector form is the same of vector starting from 0, until dim with 2 steps
    #[: (head_dim // 2)] is used subtract the last element if the head_dim is odd
    numerator = torch.arange(0, head_dim, 2)[: (head_dim // 2)].float()
    frequencies = 1.0/ (theta ** (numerator / head_dim)).to(device)


    # Position
    p = torch.arange(seq_len, device=device)

    # Multiply each theta by the position (outer product)
    # This will create a matrix (seq_len, head_dim/2)
    frequencies = torch.outer(p, frequencies).float()

    # define the same matrix with all 1
    matrix = torch.ones_like(frequencies)
    # We then have to construct the polars.
    f = torch.polar(matrix,frequencies)
    return f


def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor, device:str)-> torch.Tensor:
    """
    Apply the rotary position embedding to the input tensor.
    """
    # Reshape the input tensor to a complex tensor
    # (B, Seq_Len, Head_Dim) -> (B, Seq_Len, Head_Dim/2, 2)
    # x.shape[:-1] is the batch and the sequence length (get the first two dimensions of the tensor)
    # -1 is used as a placeholder that will automatically be calculated based on the size of the tensor and the remaining dimensions
    # 2 is the last dimension of the tensor. We will get a vector with size 2 for each element in the tensor
    # view_as_complex() will convert the tensor to a complex tensor --> The 2 elements specified before,
    # will be used as the real and imaginary part of the complex number
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Multiply the input tensor by the frequency tensor to apply the rotary position embedding
    # Final shape will be (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex#freqs_cis
    # Convert again to real tensor
    x_out = torch.view_as_real(x_rotated)
    # Reshape the tensor to the original shape
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    # Convert the tensor to the original type and move it to the original device
    return x_out.type_as(x).to(device)


class DecoderMultiHeadAttention(nn.Module):
    """
    Also known as Casual Self Attention.
    This is the attention head that will be used in the decoder.
    The attention head will take the input and compute the attention weights
    based on the query, key, and value.
    The future tokens are masked so that the attention head only looks at 
    the previous tokens (this is what makes it a decoder attention head).
    """
    def __init__(self, config:ModelConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        # Instead of defining the linear layers for the query, key, and value separately
        # we define a single linear layer that will output the query, key, and value.
        # This is done to improve the performance of the model.
        self.att = nn.Linear(config.n_embd, config.n_embd*3, bias=False)
        self.n_head = config.n_head
        # Register buffer as it is not a parameter. The ones with the tril (lower triangular) matrix with 1s
        # is used to mask the upper triangular part of the matrix. We just want the attention to look at the
        # previous tokens and not the future tokens. This will aggregate the means of the previous tokens.
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.linear_projection = nn.Linear(config.n_embd, config.n_embd)
        self.projection_dropout = nn.Dropout(config.dropout)
        # Add dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(self,x:torch.Tensor, rope_freqs:torch.Tensor):
        # batch is the size of the batch
        # tokens is the number of tokens in the sequence
        # head_size is the size of the hidden layer in the attention head
        B,T,C = x.shape # batch, tokens, channels
        # Split the output of the linear layer into query, key, and value
        # The output of the linear layer will be of size (B, T, C)
        k,q,v = self.att(x).chunk(3, dim=-1)
        # Modify each vector to fit the shape including the head size
        k = k.view(B, T, self.n_head,C//self.n_head ).transpose(1,2) # batch, head size, tokens, channels // head size
        q = q.view(B, T, self.n_head,C//self.n_head ).transpose(1,2) # batch, head size, tokens, channels // head size
        v = v.view(B, T, self.n_head,C//self.n_head ).transpose(1,2) # batch, head size, tokens, channels // head size


        # Apply the rotary position embedding
        k = apply_rope(k, rope_freqs, x.device)
        q = apply_rope(v, rope_freqs, x.device)

        if not self.flash:
            # Don't apply custom mask as the param is_causal already apply the mask
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
            # batch, head_size, tokens, channels // head_size
        else:

            # Compute the attention weights
            # k.transpose(2,3) --> batch, head_size, tokens, channels // head_size
            # q @ k.transpose(2,3) will compute the dot product of the query and the key.-> batch, head_size, tokens, tokens
            # Here is where the magic happens. We compute the attention weights by taking the dot product of the query
            # and the key. We then divide by the square root of the head size. This is the scaled dot product attention. 
            wei = (q @ k.transpose(2,3)) * (k.shape[-1] ** -0.5) # batch, head_size, tokens, tokens
            # we have to mask the upper triangular part of the matrix
            # we do this by adding a large negative number to the places we want to mask
            # and then take the softmax of the matrix. 
            # This is done so that the softmax of the upper triangular part of the matrix is 0 when softmaxed
            # and the other parts are not affected.
            # self.tril[:T, :T] == 0 will return a matrix of the same size as wei with 1s (True) in the upper triangular part
            # masked_fill() will replace the 1s (True) with -inf.
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # batch, head_size, tokens, tokens

            # apply softmax and dropout
            # apply the softmax to the last dimension of the matrix
            wei = F.softmax(wei, dim=-1) # batch, head_size, tokens, tokens
            wei = self.attn_dropout(wei)

            # This means that values will pass through the attention head and the output will be the weighted sum of the values
            # based on the attention weights. 
            output = wei @ v # batch, head_size, tokens, channels // head_size
        # First we need to reshape again the output of the attention head
        # Reshape the output of the attention head to the original shape
        # (B, T, H, C//H) -> (B, T, C)
        # Use contiguous() to make sure that the tensor is stored in a contiguous block of memory
        output = output.transpose(1,2).contiguous().view(B, T, C)
        # Apply the linear projection to combine the output of the attention heads
        output = self.projection_dropout(self.linear_projection(output))
        return output



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
        self.attn = DecoderMultiHeadAttention(config)
        self.ffn = FFN(config)
        
        # RoPE
        # We need to initialize the frequency tensor for the rotary position embedding
        self.rope_frequencies = _rope_frequency(config.n_embd//2, config.block_size, device=config.device)
    
    def forward(self,x:torch.Tensor ):
        B, Seq_len, C = x.shape
        # Take into account that we apply the normalization before the attention block
        # This is a modification from original paper Attention is All You Need
        # (a better implementation)
        rope_freqs = self.rope_frequencies#[0:Seq_len]
        x = x + self.attn(self.ln1(x), rope_freqs)
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
        # DEPRECATED: self.positional_embedding = nn.Embedding(config.block_size, config.n_embd)
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
        B, Seq_len = idx.shape
        # Get the device of the input
        device = idx.device
        assert Seq_len <= self.block_size, f"Cannot forward sequence of length {Seq_len}, block size is only {self.block_size}"

        # idx and targets are both (B,T) tensor of integers
        x = self.token_embedding(idx) # (B,Seq_len, C)
        # pos_emb = self.positional_embedding(torch.arange(T, device=device)) # (T,C)
        # DEPRECATED (we use RoPE Now): Add the positional embedding to the token embedding
        # x = x + pos_emb # (B,Seq_len,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        
        # This is for training the model. We will compute the loss
        # by comparing the logits with the targets.
        if targets is None:
            loss = None
        else:
            B, Seq_len, C = logits.shape
            # Flatten the logits and the targets
            logits = logits.view(B*Seq_len, C)
            targets = targets.view(B*Seq_len)
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
