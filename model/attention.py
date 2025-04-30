import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, config, vis=False):
        super(Attention, self).__init__()
        assert config.hidden_size % config.transformer["num_heads"] == 0, "hidden_size must be divisible by num_heads"
        self.vis = vis
        self.num_heads = config.transformer["num_heads"]
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.all_head_dim = self.num_heads * self.head_dim

        self.query_dense = nn.Linear(self.hidden_size, self.all_head_dim)
        self.key_dense = nn.Linear(self.hidden_size, self.all_head_dim)
        self.value_dense = nn.Linear(self.hidden_size, self.all_head_dim)
        
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.output_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = nn.Softmax(dim=-1)
    
    def transpose_for_score(self, x):
        # x: (B, N, all_head_dim) -> (B, num_heads, N, head_dim)
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
     
    def forward(self, x):
        B, N, C = x.shape

        query = self.query_dense(x)
        key = self.key_dense(x)
        value = self.value_dense(x)

        query = self.transpose_for_score(query)
        key = self.transpose_for_score(key)
        value = self.transpose_for_score(value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        # Context (Attention Value)
        context = torch.matmul(attention_probs, value).permute(0, 2, 1, 3).contiguous().reshape(B, N, C)

        output = self.output_dense(context)
        output = self.proj_dropout(output)

        if self.vis:
            return output, attention_probs
        else:
            return output