from torch import nn
from models.layers.multihead_attention import MultiHeadAttention
from models.layers.norm import LayerNorm
from models.layers.position_wise_feedforward import PositionWiseFeedForward

class Encoder(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, p_drop=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(p=p_drop)
        self.norm1 = LayerNorm(d_model)
        
        self.feed_forward = PositionWiseFeedForward(d_model, ffn_hidden)
        self.dropout2 = nn.Dropout(p=p_drop)
        self.norm2 = LayerNorm(d_model)
        
    def forward(self, src, mask=None):        
        # Multi-head Attention 
        x = self.attention(query=src, key=src, value=src, mask=mask)
        x = self.dropout1(x)
        
        # Add & Norm
        x = self.norm1(src + x)
        
        # Feed Forward
        feed_x = self.feed_forward(x)
        feed_x = self.dropout2(feed_x)
        
        # Add & Norms
        x = self.norm2(feed_x + x)
        
        return x