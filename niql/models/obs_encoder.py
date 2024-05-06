import torch
import torch.nn as nn
import torch.nn.functional as F


class ObservationEmbeddingModel(nn.Module):
    def __init__(self, input_dim, num_heads, device, dropout=0.1):
        super(ObservationEmbeddingModel, self).__init__()
        assert input_dim % num_heads == 0, f"Input dimension {input_dim} must be divisible by the number of heads {num_heads}"
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.W_o = nn.Linear(input_dim, input_dim)
        self.W_agg = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, x):
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
        x = self.W_o(x)

        # Aggregation
        x = torch.sum(x, dim=1)  # gather across the neighbour dimension
        x = self.W_agg(x)
        x = F.elu(x)

        return x
