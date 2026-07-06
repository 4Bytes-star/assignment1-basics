import torch
import torch.nn as nn
import math
import numpy as np
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import einsum, rearrange

class linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module. This function should accept the following parameters:
            in_features: int  final dimension of the input
            out_features: int  final dimension of the output
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        # call superclass constructor
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1.construct the parameters
        self.W = nn.Parameter(torch.empty((out_features, in_features),device=device, dtype=dtype))
        # self.b = nn.Parameters(torch.zeros(out_features, device=device, dtype=dtype)

        # 2.initialize the parameters
        std = math.sqrt(2.0 / (in_features + out_features))

        # 3.use trunc_normal to initialize 
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3.0*std, b=3*std)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        # implement y = xW^t (+b)
        # x: (..., in_features)
        # W: (out_features, in_features)
        # y: (..., out_features)
        # return x @ self.W.t() # + self.b
        return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")

class embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module. This function should accept the following parameters:
            num_embeddings: int  Size of the vocabulary
            embedding_dim: int  Dimension of the embedding vectors, i.e., 𝑑model
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 1.Construct the parameters using nn.Parameters
        self.matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        # 2.Initialize the parameters
        std = 1
        
        # 3.use the trunc_normal_ to initialize
        nn.init.trunc_normal_(self.matrix, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        """
        # 核心逻辑：利用 PyTorch 的索引功能
        # 当我们使用 tensor 对另一个 tensor 进行索引时，
        # 输出的形状将是 索引形状 + 被索引张量剩余的维度
        # 这里 token_ids 是 (batch_size, sequence_length)
        # self.matrix 是 (vocab_size, embedding_dim)
        # 因此结果会自动变为 (batch_size, sequence_length, embedding_dim)
        return self.matrix[token_ids]
    
class rmsnorm(nn.Module):
    def __init__(self, d_model:int, eps:float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
            d_model: int  Hidden dimension of the model
            eps: float = 1e-5  Epsilon value for numerical stability
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Construct the parameters that needs
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        # 将输入转为 float32 以保证数值稳定性（RMS计算涉及平方和开方）
        x = x.to(torch.float32)
        
        # 1. 计算均方根 (RMS)
        # x.pow(2) 对每个元素平方
        # .mean(-1, keepdim=True) 在最后一个维度（d_model）求平均值
        # + self.eps 防止除以零
        # .sqrt() 开根号
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
        # 2. 归一化并应用缩放因子 (gain)
        # 这里的除法会自动利用广播机制 (broadcasting)
        # self.gain 是 (d_model,)，会自动作用到 (batch_size, seq_len, d_model) 的每个位置
        result = (x / rms) * self.gain
        
        return result.to(in_dtype)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = linear(d_model, d_ff)
        self.w2 = linear(d_ff, d_model)
        self.w3 = linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> Float[Tensor, "... d_model"]:
        # FFN(𝑥) = SwiGLU(𝑥, 𝑊1, 𝑊2, 𝑊3) = 𝑊2(SiLU(𝑊1𝑥) ⊙ 𝑊3𝑥)
        hidden1 = self.w1(x)
        
        hidden3 = self.w3(x)
        
        swish_out = hidden1 * torch.sigmoid(hidden1)
        
        return self.w2(swish_out * hidden3)
        
        
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # 1. Construct the freqs: [d_k/2]
        # formula: theta ^ (-2i / d_k),i: 0~d_2-1
        i = torch.arange(0, d_k, 2, device=device).float()
        freqs = 1.0 / (theta ** (i / d_k))
        
        # 2.cal the max seq position cos ans sin,then cache the results
        max_token_position = torch.arange(max_seq_len)
        cached_position_angles = max_token_position.unsqueeze(-1) * freqs.to(device)
        self.cached_cos = torch.cos(cached_position_angles)
        self.cached_sin = torch.sin(cached_position_angles)
        
        # 3.使用register_buffer保存状态, 可随设备迁移
        self.register_buffer("cos_cached", self.cached_cos, persistent=False)
        self.register_buffer("sin_cached", self.cached_sin, persistent=False)
        
    def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
        # x shape ： (..., seq_len, d_k)
        # token_position shape: (..., seq_len)
        # 2. Compute the angles: (..., seq_len, d_k/2)
        # use the pytorch broadcast: (..., seq_len, 1) * (d_k / 2) -> (..., seq_len, d_k / 2)        
        if token_position is None:
            token_position = torch.arange(x.shape[-2])
        # angles = token_position.unsqueeze(-1) * self.freqs.to(token_position.device)

        # 3.Compute cos and sin
        # cos = torch.cos(angles)
        # sin = torch.sin(angles)
        # 已经缓存了cos 和sin，直接索引即可
        selected_cos = self.cached_cos[token_position]
        selected_sin = self.cached_sin[token_position]

        # 4.split the input tensor
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        # 5.rotate 
        # x_{2i} = x_{2i} * cos - x{2i+1} * sin
        # x_{2i+1} = x_{2i} * sin + x{2i+1} * cos
        x_prime_even = x_even * selected_cos - x_odd * selected_sin
        x_prime_odd = x_even * selected_sin + x_odd * selected_cos
        
        out = torch.empty_like(x)
        out[..., ::2] = x_prime_even
        out[..., 1::2] = x_prime_odd
        
        return out
        
        
class multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被n_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # define the Q,K,V matrix together
        self.qkv_proj = linear(d_model, 3 * d_model)
        
        self.out_proj = linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor):
        # x: [batch, seq_len, d_model]
        batch, seq_len, _ = x.shape
        
        # 1.cal the Q,K,V
        qkv = self.qkv_proj(x)

        # 2.rearrange 
        qkv = rearrange(qkv, 'b s (three h d) -> three b h s d', three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Q = rearrange(Q, 'b s (h d) -> b h s d',h = self.num_heads)
        # K = rearrange(K, 'b s (h d) -> b h s d',h = self.num_heads)
        # V = rearrange(V, 'b s (h d) -> b h s d',h = self.num_heads)

        # 3.implement the casual mask
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        mask = torch.tril(mask)

        from cs336_basics.nn_utils import scaled_dot_product_attention
        output = scaled_dot_product_attention(q, k, v, mask)
        output = rearrange(output, 'b h s d -> b s (h d)')
        
        return self.out_proj(output)

        
class multihead_self_attention_with_rope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, RoPE: RotaryPositionalEmbedding):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被n_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # define the Q,K,V matrix together
        self.qkv_proj = linear(d_model, 3 * d_model)        
        self.out_proj = linear(d_model, d_model)
        self.RoPE:RotaryPositionalEmbedding = RoPE
    
    def forward(self, x: torch.Tensor, token_positions: Int[Tensor, " ... sequence_length"] | None = None):
        batch, seq_len, _ = x.shape

        # 1.cal the Q,K,V
        qkv = self.qkv_proj(x)

        # 2.rearrange
        qkv = rearrange(qkv, 'b s (three h d) -> three b h s d', three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 3.implement the casual mask
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        mask = torch.tril(mask)
        from cs336_basics.nn_utils import scaled_dot_product_attention
        q = self.RoPE(q, token_positions)
        k = self.RoPE(k, token_positions)

        output = scaled_dot_product_attention(q, k, v, mask)
        output = rearrange(output, 'b h s d -> b s (h d)')

        return self.out_proj(output)
    
class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):        
        super().__init__()
        self.rmsnorm1 = rmsnorm(d_model)
        self.rmsnorm2 = rmsnorm(d_model)
        rope = RotaryPositionalEmbedding(theta, d_model//num_heads, max_seq_len)
        self.casual_multihead_self_attn = multihead_self_attention_with_rope(d_model, num_heads, rope)

        self.ffn = FeedForward(d_model, d_ff)
    
    def forward(self, x: torch.Tensor) -> Float[Tensor, " batch_size seq_len d_model"]:
        y = x + self.casual_multihead_self_attn(self.rmsnorm1(x))

        z = y + self.ffn(self.rmsnorm2(y))

        return z

class transformer_lm(nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        d_model: int, 
        num_layers: int,
        num_heads: int, 
        d_ff: int, 
        rope_theta: float,
    ):
        super().__init__()
        self.embedding = embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=rope_theta, max_seq_len=context_length))
        self.norm = rmsnorm(d_model=d_model)
        self.lm_head = linear(d_model, vocab_size)

    def forward(self, in_indices: Int[Tensor, " batch_size seq_len"]) -> Float[Tensor, " batch_size seq_len vocab_size"]:
        # 1.先经过embedding
        x = self.embedding(in_indices)
        # after this the shape is (batch_size seq_len d_model)
        
        # 2.进入transformer block
        for layer in self.layers:
            x = layer(x)
        
        # 3.出来后经过一个norm
        y = self.norm(x)
        
        z_probs = self.lm_head(y)
        # from cs336_basics.nn_utils import softmax
        # z_probs = softmax(z_probs, dim=-1)
        
        return z_probs