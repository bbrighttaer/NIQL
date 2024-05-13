import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttentionEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, device, dropout=0.1):
        super(MultiHeadSelfAttentionEncoder, self).__init__()
        assert input_dim % num_heads == 0, f"Input dimension {input_dim} must be divisible by the number of heads {num_heads}"
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.latent_dim = 128

        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.W_enc = nn.Linear(input_dim, self.latent_dim)
        self.W_out = nn.Linear(self.latent_dim, input_dim)
        self.ste = StraightThroughEstimator()

        self.dropout = nn.Dropout(dropout)
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)  # gather across the neighbour dimension
        query, key, value = x, x, x
        batch_size = query.shape[0]

        # Linear transformation for query, key, and value
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale_factor

        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(energy, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to value
        x = torch.matmul(attention_weights, V)

        # Concatenate heads and apply final linear transformation
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.input_dim)
        enc = self.ste(self.W_enc(x))
        out = F.relu(self.W_out(enc))
        return out, enc


class FCNEncoder(nn.Module):

    def __init__(self, input_dim, num_tiling=3, **kwargs):
        super().__init__()
        self.encoding_dim = 8
        self.ste = StraightThroughEstimator()
        self.lin_projection = nn.Linear(input_dim, input_dim)
        self.tiling_id = torch.eye(num_tiling).to(kwargs.get("device", "cpu"))

        # create tilings
        self.tilings = nn.ModuleList([nn.Linear(input_dim + num_tiling, self.encoding_dim) for _ in range(num_tiling)])

        # reduction layer
        self.lin_reduce = nn.Linear(self.encoding_dim, input_dim)

    def forward(self, x):
        # projection
        x = F.elu(self.lin_projection(x))

        # encoding
        encoding = []
        for i, tiling in enumerate(self.tilings):
            t_id = self.tiling_id[i]
            t_id = t_id.view(1, 1, -1).expand(x.shape[:2] + (-1,))
            t_x = torch.cat([x, t_id], dim=-1)
            enc = self.ste(tiling(t_x))
            encoding.append(enc)
        encoding = torch.mean(torch.stack(encoding), dim=0, keepdim=True)

        # res-link
        x = F.elu(self.lin_reduce(encoding) + x)
        return x, encoding


class STEFunction(torch.autograd.Function):
    """
    Credit: https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0
    """

    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # passed through hard-tanh to clip gradients between [-1, 1]
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x
