import math
from utils.base import *
from utils.common import *
    
##实现编码器单层，并添加前馈层
class EncoderLayer(nn.Module):
    def __init__(self, model_dim, n_head, feedforward_dim, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head, model_dim)

        ## 前馈层，feed forward net (FFN)
        self.feedforward = nn.Sequential(nn.Linear(model_dim, feedforward_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(feedforward_dim, model_dim),
                                         nn.Dropout(dropout, inplace=True))
        self.layernorm = nn.LayerNorm(model_dim)
    
    def forward(self, src, src_mask = None):
        src = self.self_attn(src, src, src, src_mask)   #前文已实现残差连接并进行了归一化
        ## FFN残差连接
        src = self.layernorm(src + self.feedforward(src))
        return src

##实现位置编码
class PositionalEncoder(nn.Module):
    def __init__(self, model_dim, max_len = 256, ):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1) #(max_len, 1)

        #写为指数形式方便计算
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(1e4)/model_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #(1, max_len, model_dim)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        ##输入x为嵌入矩阵 (batch_size, seq_len, model_dim)
        x = x + self.pe[:, :x.size(1), :]
        return x