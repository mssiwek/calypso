import inspect
import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    """
    A Long Short-Term Memory (LSTM) neural network for time-series regression.

    Attributes:
        ...
    """
    def __init__(self, input_size, output_channels=1, \
                 max_len=100, latent_dim=32, \
                 hidden_size=64, num_layers=5, \
                 dropout=0.0, act=None, **kwargs):
        """
            output_channels: number of outputs, i.e. either just mean or mean AND vairance
        """
        
        self.act = act

        # add all arguments to self
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for arg in args[1:]:  # Skip 'self'
            setattr(self, arg, values[arg])
        
        # Save init kwargs for re-instantiating later
        self.init_kwargs = {arg: values[arg] for arg in args[1:]}

        super(LSTMRegressor, self).__init__()
        
        self.latent = nn.Linear(input_size, latent_dim)

        # time embeddings
        self.time_embedding = nn.Embedding(max_len, latent_dim)

        self.lstm = nn.LSTM(input_size=latent_dim*2, \
                            hidden_size=hidden_size, \
                            num_layers=num_layers, \
                            batch_first=True, dropout=dropout)
        
        self.fc = nn.Linear(hidden_size, output_channels)



    def activation(self):
        """
            Activation functions to map outputs and introduce non-linearity. 
            Examples:
            'Sigmoid'
            'Tanh' (Hyperbolic Tangent)
            'ReLU' (Rectified Linear Unit)
            'ELU' (Exponential Linear Unit)
        """
        if self.act == 'Sigmoid':
            return(nn.Sigmoid())
        if self.act == 'Tanh':
            return(nn.Tanh())
        if self.act == 'ReLU':
            return(nn.ReLU())
        if self.act == 'ELU':
            return(nn.ELU())

    def forward(self, input, sequence_length=100, hidden=None):
        """
        Parameters
        ----------
        x : Tensor, shape (B, 2)
            Binary parameters (e_b, q_b).
        sequence_length : int
            Number of timesteps in the output sequence.

        Returns
        -------
        Tensor of shape (B, T) or (B, T, output_channels)
            Lightcurve prediction for each timestep.
            B: batch size
            T: sequence length
        """
        device = input.device
        B = input.shape[0]

        # Encode (e_b, q_b) → latent
        z = self.latent(input)  # (B, latent_dim)
        z = z.unsqueeze(1) 
        # Create time indices [0, 1, ..., T-1]
        t_idx = torch.arange(sequence_length, device=device).unsqueeze(0).repeat(B, 1)  # (B, T)
        # Get time embeddings
        t_emb = self.time_embedding(t_idx)  # (B, T, latent_dim)
        
        # OLD: Add time signal to latent z → (B, T, latent_dim)
        # z_seq = z + t_emb
        
        # NEW: Concatenate time signal to latent z → (B, T, latent_dim*2)
        z_seq = torch.cat([t_emb, z], dim=-1)  # (B, T, 2*latent_dim)

        # LSTM decoding
        lstm_out, hidden = self.lstm(z_seq)  # (B, T, hidden_dim)

        # Project to accretion rate at each timestep
        out = self.fc(lstm_out)  # (B, T, output_channels)
        
        if self.act is not None:
            print("self.act = ", self.act)
            out = self.activation()(out)
        
        return(out, hidden)



