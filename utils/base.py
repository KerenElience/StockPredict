from common import *

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(dropout, inplace=True)
                                 )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv(x)
        x += residual
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, scale_factor, dropout = 0.0):
        super().__init__()
        self.scale_factor = scale_factor #一般缩放系数为k向量维度的算数平方根
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, q, k, v, mask = None):
        ## q shape为(batch_size, n_head, len_q, dim_k) dim_k等于输入维度除以头数目，必须整除
        ## k shape为(batch_size, n_head, len_k, dim_k)
        ## v shape为(batch_size, n_head, len_v, dim_v) 通常dim_v设置于dim_k相同, len_q/k/v也是相同的等于最长句子的长度
        ## 非多头注意力时，n_head维度不存在
        attn = torch.matmul(q / self.scale_factor, k.transpose(-2,-1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(self.softmax(attn, dim = -1))
        attn = torch.matmul(attn, v) #(batch_size, n_head, len_q, dim_v)
        return attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, model_dim, dropout = 0.1):
        super().__init__()
        self.n_head = n_head
        self.model_dim = model_dim
        self.head_dim = model_dim // n_head
        assert model_dim % n_head == 0 , "输入维度要能被头的个数整除"

        ##q, k, v经过线性变换映射到对应的查询，键，值空间
        self.wq = nn.Linear(model_dim, model_dim) #第二个值应该是 n_head * head_dim
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)
        self.fc = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim) ## BatchNormlize归一化的是所有通道，LayerNorm是每个批次，前者适用于卷积图像

        self.attention = SelfAttention(scale_factor = self.head_dim**0.5)
    
    def forward(self, x1, x2, x3, mask = None):
        ##x初次输入应该是包含位置信息的嵌入矩阵 (batch_size, seq_len, dim), x1,x2,x3在编码器中是相同的矩阵
        ##根据x输入生成q, k ,v
        batch_size, seq_len, model_dim = x1.size()
        residual = x1    #残差
        q = self.wq(x1).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)
        k = self.wk(x2).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)
        v = self.wv(x3).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)

        #自注意力层输出 (batch_size, n_head, seq_len, dim) -> (batch_size, seq_len, dim)
        scores = self.attention(q, k, v, mask).transpose(1,2).contiguous().view(batch_size, seq_len, model_dim)
        ##自注意力层的残差连接 Add&Normalize
        scores = self.layer_norm(residual + self.dropout(self.fc(scores)))
        return scores