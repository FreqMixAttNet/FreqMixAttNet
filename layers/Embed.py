import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

    
    

class TokenEmbeddingGCN(nn.Module):
    
    

    def __init__(self, c_in, d_model,patch_len):
        super(TokenEmbeddingGCN, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=padding)
        self.ttcn_dim = d_model
        self.patch_len = patch_len
        input_dim = patch_len
        ttcn_dim = d_model
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.Filter_Generators = nn.Sequential(
            nn.Linear(input_dim, ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            # nn.Linear(ttcn_dim, ttcn_dim, bias=True),
            # nn.ReLU(inplace=True),
            nn.Linear(ttcn_dim, input_dim*ttcn_dim, bias=True),
            nn.ReLU(inplace=True))
            
        self.T_bias = nn.Parameter(torch.randn(1, ttcn_dim))
    def TTCN(self, X_int):
        # X_int: shape (B, N, F_in)


        B = X_int.shape[0]
        N = X_int.shape[1]
        F_in = X_int.shape[2]
        Filter = self.Filter_Generators(X_int) # (B, N, F_in*ttcn_dim)
        # print("GCN Filter:",Filter.shape,X_int.shape)
        # normalize along with sequence dimension
        Filter_seqnorm = F.softmax(Filter, dim=-2)  # (B, N, F_in*ttcn_dim)
        Filter_seqnorm = Filter_seqnorm.view(B, N, self.ttcn_dim, -1) # (B, N, ttcn_dim, F_in)
        # print("GCN Filter_seqnorm:",Filter_seqnorm.shape)
        # X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)

        
        # X_int_broad = X_int.unsqueeze(2).repeat(1, 1, F_in, 1) # (B,N,F_in,F_in)
        X_int_broad = X_int.unsqueeze(2).repeat(1, 1, self.ttcn_dim, 1) # (B,N,ttcn_dim,F_in)
        # print("GCN X_int_broad:",X_int_broad.shape)
        ttcn_out =   Filter_seqnorm * X_int_broad

        # ttcn_out = torch.matmul(Filter_seqnorm, X_int_broad)
        # ttcn_out = torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1) # (N, ttcn_dim)
        # print("GCN ttcn_out:",ttcn_out.shape)
        # print("GCN T_bias:",self.T_bias.shape)
        ttcn_out = ttcn_out.permute(0, 1, 3,2) #(B, N, F_in, ttcn_dim)
        # print("GCN ttcn_out1:",ttcn_out.shape)
        h_t = ttcn_out
        # h_t = torch.relu(ttcn_out + self.T_bias) # (N, ttcn_dim)
        # print("GCN h_t:",h_t.shape)
        h_t = h_t.reshape(B, N*F_in, -1) # (B, N*F_in, ttcn_dim)
        h_t = self.pool(h_t)
        # print("GCN h_t:",h_t.shape)

        # print("GCN h_t1:",h_t.shape)
        return h_t               
    def forward(self, x):
        # print("GCNX1:",x.shape)
        bs = x.shape[0]
        n_vars = x.shape[1]
        # Input encoding

    
        x = torch.reshape(x, (bs,-1,self.patch_len))               # x: [bs * nvars x seq_patch_num x d_model]
        # print("GCNX1:",x.shape)
        x = self.TTCN(x)
        # print("GCNX2:",x.shape)
        # x = self.TTCN(x.permute(0, 2, 1)).transpose(1, 2)
        return x    
    

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
            self.hour_embed = Embed(hour_size, d_model)
        if freq == 'h':
            self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3]) if hasattr(self, 'hour_embed') else 0.
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

class TemporalEmbedding_v2(nn.Module):
    def __init__(self, d_model, embed_type='timeH', freq='h'):
        super(TemporalEmbedding_v2, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
            self.hour_embed = Embed(hour_size, d_model)
        if freq == 'h':
            self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
        
        #self.weekofyear_embed = Embed(54, d_model)
        self.weekend_embed = Embed(2, d_model)
        self.quarter_embed = Embed(5, d_model)
        self.holidays_embed = Embed(2, d_model)
        # self.around_315_embed = Embed(2, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3]) if hasattr(self, 'hour_embed') else 0.
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        
        # weekofyear = self.weekofyear_embed(x[:, :, -4])
        is_weekend = self.weekend_embed(x[:, :, -3])
        quarter = self.quarter_embed(x[:, :, -2])
        is_holidays = self.holidays_embed(x[:, :, -1])
        # is_around_315 = self.around_315_embed(x[:, :, -1])

        return hour_x + weekday_x + day_x + month_x + minute_x + is_weekend + quarter + is_holidays # + weekofyear # + is_around_315 


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='timeH', freq='h', dropout=0.1,patch_len=32):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model,patch_len=patch_len)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if embed_type == 'timeH':
            self.temporal_embedding = TemporalEmbedding_v2(d_model=d_model, embed_type=embed_type,freq=freq)  
        elif embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,freq=freq)  
        else:
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)



class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if embed_type == 'timeH':
            self.temporal_embedding = TemporalEmbedding_v2(d_model=d_model, embed_type=embed_type,freq=freq)  
        elif embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,freq=freq)  
        else:
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x is None and x_mark is not None:
            return self.temporal_embedding(x_mark)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            emb1, emb2 = self.value_embedding(x), self.temporal_embedding(x_mark)
            # print(emb1.shape, emb2.shape)
            x = emb1+ emb2
        return self.dropout(x)

class TemporalEmbedding_v3(nn.Module):
    def __init__(self, d_model, embed_type='timeH', freq='d'):
        super(TemporalEmbedding_v3, self).__init__()
        self.freq = freq
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = nn.Embedding
        
        emb_num = 6
        if freq == 't':
            emb_num +=2
        if freq == 'h':
            emb_num +=1
        d_model1 = d_model
        emb_size = d_model1//emb_num
        print("emb_size:",emb_size,"emb_num:",emb_num)
        self.weekday_embed = Embed(weekday_size, emb_size)
        self.day_embed = Embed(day_size, emb_size + d_model1%emb_num)
        self.month_embed = Embed(month_size, emb_size)
        if freq == 't':
            self.minute_embed = Embed(4, emb_size)
            self.hour_embed = Embed(24, emb_size)
        if freq == 'h':
            self.hour_embed = Embed(24, emb_size)
        
        #self.weekofyear_embed = Embed(54, d_model)
        self.weekend_embed = Embed(2, emb_size)
        self.quarter_embed = Embed(5, emb_size)
        self.holidays_embed = Embed(2, emb_size)
        
        self.embed_layer = nn.Linear(d_model1, d_model)

    def forward(self, x):
        x = x.long()
        total_emb = []
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        
        is_weekend = self.weekend_embed(x[:, :, -3])
        quarter = self.quarter_embed(x[:, :, -2])
        is_holidays = self.holidays_embed(x[:, :, -1])
        total_emb = [weekday_x, day_x, month_x, is_weekend, quarter, is_holidays ]
        if self.freq == 't':
            hour_x = self.hour_embed(x[:, :, 3])
            min_x = self.minute_embed(x[:, :, 4])
            total_emb.append(hour_x)
            total_emb.append(min_x)
        if self.freq == 'h':
            hour_x = self.hour_embed(x[:, :, 3])
            total_emb.append(hour_x)
        # print("total_emb:","weekday_x:",weekday_x.shape,"day_x:",day_x.shape)
        emb  = torch.concat(total_emb, dim=-1) 
        
        emb = self.embed_layer(emb)
        # print("total_emb:",len(total_emb),"emb0:",emb.shape)
        return emb

    
class Gating_DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1,patch_len=32):
        super(Gating_DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbeddingGCN(c_in=c_in, d_model=d_model,patch_len=16)
        self.value_embedding1 = TokenEmbeddingGCN(c_in=c_in, d_model=d_model,patch_len=12)

        # self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # self.value_embedding1 = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if embed_type in [ 'timeH', 'timeF']:
            self.temporal_embedding = TemporalEmbedding_v3(d_model=d_model, embed_type=embed_type,freq=freq)  
        else:
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.weight = nn.Parameter( torch.randn(1, d_model)) 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x is None and x_mark is not None:
            return self.temporal_embedding(x_mark) * self.weight
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            lens =  x.shape[1]
            # emb1, emb2 = self.value_embedding(x), self.temporal_embedding(x_mark)

            if(lens>=48):
                emb1, emb2 = self.value_embedding(x), self.temporal_embedding(x_mark)
            else:
                emb1, emb2 = self.value_embedding1(x), self.temporal_embedding(x_mark)
            # x = emb1 * torch.sigmoid(1-self.weight) + emb2  * self.weight
            x = emb1 + emb2  * self.weight

            # x = emb1 * emb2
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
