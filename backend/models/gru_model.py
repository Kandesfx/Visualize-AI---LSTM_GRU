"""
GRU Sentiment Analysis Model.
Kiến trúc đơn giản 1 layer GRU cho demo trực quan.
"""
import torch
import torch.nn as nn


class GRUSentiment(nn.Module):
    """
    GRU model cho phân tích cảm xúc.
    
    Architecture:
        Embedding(vocab_size, 128) -> GRU(128, 64) -> Dropout -> Linear(64, 2)
    """
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64,
                 output_dim=2, n_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        # Layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0 if n_layers == 1 else dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        """
        Args:
            text: [batch_size, seq_len] - input token indices
        Returns:
            logits: [batch_size, output_dim]
        """
        # text: [batch, seq_len]
        embedded = self.dropout(self.embedding(text))
        # embedded: [batch, seq_len, embed_dim]
        
        output, hidden = self.gru(embedded)
        # hidden: [n_layers, batch, hidden_dim]
        
        # Lấy hidden state của layer cuối cùng
        hidden = self.dropout(hidden[-1])
        # hidden: [batch, hidden_dim]
        
        logits = self.fc(hidden)
        # logits: [batch, output_dim]
        
        return logits
