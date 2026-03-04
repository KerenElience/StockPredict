from utils.base import *

class ChoseStock(nn.Module):
    def __init__(self, in_channels, out_channels, residual_cfg, dropout = 0.1):
        super().__init__()
        self.hidden_dim = 768
        self.residual_cfg = residual_cfg
        self.dropout = dropout
        self.pre_conv = nn.Sequential(nn.BatchNorm2d(in_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, padding=1)
                                    )
        
        self.layer = self._make_residual_layer()

        self.fc = nn.Linear(out_channels, 1)
        self.softmax = nn.Softmax(dim = -1)

    def _make_residual_layer(self):
        layer = []
        in_channels = self.hidden_dim
        for x in self.residual_cfg:
            layer += [ResidualBlock(in_channels, x, self.dropout)]
            in_channels = x
        return nn.Sequential(*layer)

    def forward(self, x, seq = None):
        """
        seq_dim is 768, x shape [1, technology index, window size, 1]
        """
        if seq is None:
            seq = torch.zeros([1, 768, 1, 1])
        else:
            seq
        x = self.pre_conv(x) #[1, 768, window_size, 1]
        x = torch.cat([x, seq], dim = -1)
        x = self.layer(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.softmax(self.fc(x))
        return x