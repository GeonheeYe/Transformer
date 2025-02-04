from torch import nn
from models.layers.multihead_attention import MultiHeadAttention
from models.layers.norm import LayerNorm
from models.layers.position_wise_feedforward import PositionWiseFeedForward

class Decoder(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, p_drop=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(p=p_drop)
        self.norm1 = LayerNorm(d_model)
        
        self.encoder_decoder_attention = MultiHeadAttention(d_model, n_head)
        self.dropout2 = nn.Dropout(p=p_drop)
        self.norm2 = LayerNorm(d_model)
        
        self.feed_forward = PositionWiseFeedForward(d_model, ffn_hidden)
        self.dropout3 = nn.Dropout(p=p_drop)
        self.norm3 = LayerNorm(d_model)
        
    def forward(self, tgt, enc, src_mask, tgt_mask):
        # Masked Multi-Head Attention
        x = self.attention(query=tgt, key=tgt, value=tgt, mask=tgt_mask)
        x = self.dropout1(x)
    
        # Add & Norm
        x = self.norm1(x + tgt)
        
        # Multi-Head Attention
        _x = self.encoder_decoder_attention(query=x, key=enc, value=enc, mask=src_mask)
        _x = self.dropout2(_x)
        
        # Add & Norm
        x = self.norm2(_x + x)
        
        # Feed forward 
        feed_x = self.feed_forward(x)
        feed_x = self.dropout3(feed_x)
        
        # Add & Norm
        x = self.norm3(feed_x + x) 
        return x

class DecoderOnly(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, p_drop=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(p=p_drop)
        self.norm1 = LayerNorm(d_model)
        
        self.feed_forward = PositionWiseFeedForward(d_model, ffn_hidden)
        self.dropout2 = nn.Dropout(p=p_drop)
        self.norm2 = LayerNorm(d_model)
        
    def forward(self, tgt, tgt_mask):
        # Masked Multi-Head Attention
        x = self.attention(query=tgt, key=tgt, value=tgt, mask=tgt_mask)
        x = self.dropout1(x)
    
        # Add & Norm
        x = self.norm1(x + tgt)
        
        # Feed forward 
        feed_x = self.feed_forward(x)
        feed_x = self.dropout2(feed_x)
        
        # Add & Norm
        x = self.norm2(feed_x + x) 
        return x