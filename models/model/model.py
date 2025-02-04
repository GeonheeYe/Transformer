from torch import nn
from models.encoding.positional_encoding import PositionalEncoding
from models.model.encoder import Encoder
from models.model.decoder import Decoder, DecoderOnly


class Transformer(nn.Module):
    def __init__(self, d_model, max_len, temperature, n_head, ffn_hidden, p_drop, n_layers, tokenizer, device):
        super().__init__()    
        # embedding
        self.src_embedding = nn.Embedding(tokenizer.vocab_size, d_model, padding_idx=tokenizer.pad_token_id)
        self.tgt_embedding = nn.Embedding(tokenizer.vocab_size, d_model, padding_idx=tokenizer.pad_token_id)
        self.dropout1 = nn.Dropout(p=p_drop)
        self.dropout2 = nn.Dropout(p=p_drop)

        # positional_encoding
        self.positional_encoding = PositionalEncoding(
                                                        d_model=d_model,
                                                        max_len=max_len, 
                                                        temperature=temperature, 
                                                        device=device, 
                                                        )
        self.Encoder = nn.ModuleList([Encoder(
                                                d_model=d_model,
                                                n_head=n_head,
                                                ffn_hidden=ffn_hidden,
                                                p_drop=p_drop,
                                                ) 
                                      for _ in range(n_layers)])
        self.Decoder = nn.ModuleList([Decoder(
                                                d_model=d_model,
                                                n_head=n_head,
                                                ffn_hidden=ffn_hidden,
                                                p_drop=p_drop,
                                                )
                                     for _ in range(n_layers)])

        self.Linear = nn.Linear(d_model, tokenizer.vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # embedding
        em_src = self.src_embedding(src)
        em_tgt = self.tgt_embedding(tgt)

        # positional_encoding
        src_pos = self.positional_encoding(src)
        tgt_pos = self.positional_encoding(tgt)
        
        encoder_out = em_src + src_pos
        encoder_out = self.dropout1(encoder_out)
        for encoder in self.Encoder:
            encoder_out = encoder(
                                    src=encoder_out, 
                                    mask=src_mask
                                    )

        decoder_out = em_tgt + tgt_pos
        decoder_out = self.dropout2(decoder_out)
        for decoder in self.Decoder:
            decoder_out = decoder(
                                    tgt=decoder_out,
                                    enc=encoder_out,
                                    src_mask=src_mask,
                                    tgt_mask=tgt_mask,
                                    )

        out = self.Linear(decoder_out)
        return out 


# Decoder only model
class DecoderOnlyModel(nn.Module):
    def __init__(self, d_model, max_len, temperature, n_head, ffn_hidden, p_drop, n_layers, tokenizer, device):
        super().__init__()    
        # embedding
        self.tgt_embedding = nn.Embedding(tokenizer.vocab_size, d_model, padding_idx=tokenizer.pad_token_id)
        self.dropout1 = nn.Dropout(p=p_drop)

        # positional_encoding
        self.positional_encoding = PositionalEncoding(
                                                        d_model=d_model,
                                                        max_len=max_len, 
                                                        temperature=temperature, 
                                                        device=device, 
                                                        )

        self.Decoder = nn.ModuleList([DecoderOnly(
                                                d_model=d_model,
                                                n_head=n_head,
                                                ffn_hidden=ffn_hidden,
                                                p_drop=p_drop,
                                                )
                                     for _ in range(n_layers)])

        self.Linear = nn.Linear(d_model, tokenizer.vocab_size)

    def forward(self, tgt, tgt_mask):
        # embedding
        em_tgt = self.tgt_embedding(tgt)  

        # positional_encoding
        tgt_pos = self.positional_encoding(tgt)

        decoder_out = em_tgt + tgt_pos
        decoder_out = self.dropout1(decoder_out)
        for i, decoder in enumerate(self.Decoder):
            decoder_out = decoder(
                                    tgt=decoder_out,
                                    tgt_mask=tgt_mask,
                                    )

        out = self.Linear(decoder_out)
        return out 