import torch
from einops import einsum
from torch import Tensor
from jaxtyping import Float, Bool

def softmax(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax to the i-th dimension of a tensor.
    
    Args:
        tensor: The input PyTorch tensor.
        dim: The dimension along which to compute the softmax.
        
    Returns:
        A tensor of the same shape with normalized probabilities.
    """
    # 1. find the biggest element of the designated dimension
    # keep_dim = True facilitates subsequent broadcasting subtraction with the original tensor
    max_val = torch.max(tensor, dim=dim, keepdim=True)[0]

    # 2. minus the max value
    shifted_tensor = tensor - max_val
    
    # 3.calculate the exp
    exp_tensor = torch.exp(shifted_tensor)

    # 4.cal the sum
    sum_exp = torch.sum(exp_tensor, dim=dim, keepdim=True)

    # 5.return the probability distribution
    return exp_tensor / sum_exp
    
    
def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"], 
    K: Float[Tensor, "... keys d_k"], 
    V: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, "... queries d_v"]:
    """
    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.size(-1)
    
    # 1.cal Q @ K^T to get the attention scores
    # use transpose(-2, -1) to exchange the last two dimension
    #scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    scores = einsum(Q, K,"... queries d_k, ... keys d_k -> ... queries keys") / (d_k ** 0.5)
   
    # 2.apply the mask
    if mask is not None:
        scores = scores.masked_fill(mask==False, float('-inf'))
    # scores (batch_size, seqlen_q, seqlen_k)
    # 3.cal the probability weights
    attn_probs = softmax(scores, dim=-1)
    
    #output = torch.matmul(attn_probs, V)
    output = einsum(attn_probs, V, "... queries keys, ... keys d_v -> ... queries d_v")
    
    return output
