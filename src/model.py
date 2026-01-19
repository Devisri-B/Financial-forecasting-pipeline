import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Computes a weighted average of the LSTM hidden states.
    Allows the model to focus on 'critical days' in the past 60 days.
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        weights = F.softmax(self.attention(lstm_output), dim=1)
        # Context vector is the weighted sum
        context_vector = torch.sum(weights * lstm_output, dim=1)
        return context_vector, weights

class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(StockPredictor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply Attention
        attn_out, weights = self.attention(lstm_out)
        
        # Final prediction
        out = self.fc(attn_out)
        return out