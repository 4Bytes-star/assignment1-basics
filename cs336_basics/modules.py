import torch
import torch.nn as nn
import math
import numpy as np
from jaxtyping import Float
from torch import Tensor
from einops import einsum

class linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear 
        transformation module. This function should accept the following parameters:
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
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        # 1. Construct the freqs: [d_k/2]
        # formula: theta ^ (-2i / d_k),i: 0~d_2-1
        i = torch.arange(0, d_k, 2, device=device).float()
        self.freqs = 1.0 / (theta ** (i / d_k))
        
        
    def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
        # x shape ： (..., seq_len, d_k)
        # token_position shape: (..., seq_len)
        # 2. Compute the angles: (..., seq_len, d_k/2)
        # use the pytorch broadcast: (..., seq_len, 1) * (d_k / 2) -> (..., seq_len, d_k / 2)        
        angles = token_position.unsqueeze(-1) * self.freqs.to(token_position.device)

        # 3.Compute cos and sin
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # 4.split the input tensor
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        # 5.rotate 
        # x_{2i} = x_{2i} * cos - x{2i+1} * sin
        # x_{2i+1} = x_{2i} * sin + x{2i+1} * cos
        x_prime_even = x_even * cos - x_odd * sin
        x_prime_odd = x_even * sin + x_odd * cos
        
        out = torch.empty_like(x)
        out[..., ::2] = x_prime_even
        out[..., 1::2] = x_prime_odd
        
        return out
        
        