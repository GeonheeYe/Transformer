from torch import nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_head=8):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask):
        q, k, v = self.w_q(query), self.w_k(key), self.w_v(value)

        #split
        q, k, v = self.split(q), self.split(k), self.split(v)

        out = self.Scaled_Dot_Product_Attention(q, k, v, mask)
        
        # concat
        batch, n_head, length, d_model = out.shape
        
        out = out.transpose(1, 2).contiguous().view(batch, length, n_head * d_model)
        out = self.w_o(out)
        return out
    
    def split(self, tensor):
        batch, length, d_model = tensor.shape
        n_model = d_model // self.n_head
        tensor = tensor.view(batch, length, self.n_head, n_model).transpose(1, 2)
        return tensor
        
    def Scaled_Dot_Product_Attention(self, q, k, v, mask=None):
        _, _, _, n_model = q.shape
        score = torch.matmul(q, k.transpose(2,3)) / torch.sqrt(torch.tensor(n_model, dtype=torch.float32))
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
            #score = score.masked_fill(mask == 0, float('-inf'))
        
        softmax = nn.Softmax(dim=-1)
        
        score = softmax(score)
        out = torch.matmul(score, v)
        return out