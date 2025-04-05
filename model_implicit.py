from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d
import numpy as np



import torch
import numpy as np
import torch.nn as nn
# from timm.models.layers import trunc_normal_
from einops import rearrange, repeat

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU, 'approx_gelu': lambda: nn.GELU(approximate="tanh")}

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Down_Cross_Attn(nn.Module):
    def __init__(self, hidden_dim = 128, heads=8, dim_head=16, dropout=0., latent_num=64, mlp_ratio = 1):
        super().__init__()
        # act='gelu'
        act='approx_gelu'
        self.dim_head = dim_head
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.scale = dim_head ** -0.5
        self.g = latent_num
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # self.ln1 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        # self.ln2 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=1, res=False, act=act)
        self.in_project_k = nn.Linear(hidden_dim, hidden_dim,bias = False)
        self.in_project_v = nn.Linear(hidden_dim, hidden_dim,bias = False)
        self.q_weights = nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros((1,self.heads, self.g, self.dim_head)), mean=0, std=0.02, a=-2, b=2))
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )


    def forward(self, x, c):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        ca_x = modulate(self.norm1(x), shift_msa, scale_msa)
        # ca_x = self.ln1(x)
        ca_q = self.q_weights
        ca_k = self.in_project_k(ca_x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() #B H N D
        ca_v = self.in_project_v(ca_x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() #B H N D
        ca_s = self.softmax(torch.matmul(ca_q,ca_k.permute(0,1,3,2))*self.scale)  # B H G N
        ca_a = torch.matmul(ca_s, ca_v) # B H G D
        ca_a = rearrange(ca_a, 'b h g d -> b g (h d)') #B G C
        ca_a = gate_msa.unsqueeze(1) *self.to_out(ca_a)
        ca_out = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(ca_a), shift_mlp, scale_mlp)) + ca_a

        return ca_out

class Latent_Attn(nn.Module):
    def __init__(self, hidden_dim = 128, heads=8, dim_head=16, dropout=0., latent_num=64, mlp_ratio = 1):
        super().__init__()
        # act='gelu'
        act='approx_gelu'
        self.dim_head = dim_head
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.scale = dim_head ** -0.5
        self.g = latent_num
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # self.ln1 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        # self.ln2 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=1, res=False, act=act)
        self.to_q = nn.Linear(hidden_dim, hidden_dim,bias = False)
        self.to_k = nn.Linear(hidden_dim, hidden_dim,bias = False)
        self.to_v = nn.Linear(hidden_dim, hidden_dim,bias = False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        sa_x = modulate(self.norm1(x), shift_msa, scale_msa)
        # sa_x = self.ln1(x)
        sa_q = self.to_q(sa_x).reshape(B, self.g, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() #B H G D
        sa_k = self.to_k(sa_x).reshape(B, self.g, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() #B H G D
        sa_v = self.to_v(sa_x).reshape(B, self.g, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() #B H G D
        sa_s = self.softmax(torch.matmul(sa_q, sa_k.permute(0,1,3,2)) * self.scale)
        sa_s = self.dropout(sa_s)
        sa_a = torch.matmul(sa_s, sa_v)  # B H G D
        sa_a = rearrange(sa_a, 'b h g d -> b g (h d)')
        sa_out = gate_msa.unsqueeze(1) * self.to_out(sa_a)
        sa_out = sa_out + x
        out_x = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(sa_out), shift_mlp, scale_mlp)) + sa_out

        return out_x
    
class Up_Cross_Attn(nn.Module):
    def __init__(self, hidden_dim = 128, heads=8, dim_head=16, dropout=0., latent_num = 64, mlp_ratio = 1):
        super().__init__()
        # act='gelu'
        act='approx_gelu'
        self.dim_head = dim_head
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.g = latent_num
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # self.ln1 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        # self.ln2 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=1, res=False, act=act)
        self.in_project_k = nn.Linear(hidden_dim, hidden_dim,bias = False)
        self.in_project_v = nn.Linear(hidden_dim, hidden_dim,bias = False)
        self.in_project_q = nn.Linear(hidden_dim, hidden_dim,bias = False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )


    def forward(self, x, x_input, c):
        B, G, C = x.shape
        B, N, C = x_input.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        ca_x = modulate(self.norm1(x), shift_msa, scale_msa)
        # ca_x = self.ln1(x)
        ca_q = self.in_project_q(x_input).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() #B H N D
        ca_k = self.in_project_k(ca_x).reshape(B, G, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() #B H G D
        ca_v = self.in_project_v(ca_x).reshape(B, G, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() #B H G D
        ca_s = self.softmax(torch.matmul(ca_q,ca_k.permute(0,1,3,2))*self.scale)  # B H N G
        ca_a = torch.matmul(ca_s, ca_v) # B H N D
        ca_a = rearrange(ca_a, 'b h n d -> b n (h d)') #B N C
        ca_a = gate_msa.unsqueeze(1) * self.to_out(ca_a)
        ca_a = ca_a + x_input
        out_x = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(ca_a), shift_mlp, scale_mlp)) + ca_a
        return out_x


class Attention_Model(nn.Module):
    def __init__(self,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 latent_num=32,
                 ):
        super(Attention_Model, self).__init__()

        self.down_cross_attn_block1 = Down_Cross_Attn(hidden_dim = n_hidden , heads=n_head, dim_head=n_hidden//n_head,
                                                    dropout=dropout, latent_num=latent_num, mlp_ratio = mlp_ratio)
        
        self.latent_attn_blocks1 = nn.ModuleList([Latent_Attn(hidden_dim = n_hidden , heads=n_head, dim_head=n_hidden//n_head,
                                            dropout=dropout, latent_num=latent_num, mlp_ratio = mlp_ratio) 
                                            for i in range(n_layers)])
        self.up_cross_attn_block1 = Up_Cross_Attn(hidden_dim = n_hidden , heads=n_head, dim_head=n_hidden//n_head, dropout=dropout,
                                                 latent_num = latent_num, mlp_ratio = mlp_ratio)  
        

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=0.02)
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.down_cross_attn_block1.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.down_cross_attn_block1.adaLN_modulation[-1].bias, 0)        
        for block in self.latent_attn_blocks1:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.up_cross_attn_block1.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.up_cross_attn_block1.adaLN_modulation[-1].bias, 0)  

    def forward(self, x, c):
        x_input = x
        x = self.down_cross_attn_block1(x, c)

        for i in range(len(self.latent_attn_blocks1)):
            x = self.latent_attn_blocks1[i](x, c)

        x = self.up_cross_attn_block1(x, x_input, c)   
        return x





class Preprocessing_Model(nn.Module):
    def __init__(self,
                 input_dim=3,
                 n_hidden=512,
                 act='gelu',
                 k = 10
                 ):
        super(Preprocessing_Model, self).__init__()

        self.preprocess = MLP(input_dim, n_hidden, n_hidden, n_layers=0, res=False, act=act)
        # self.preprocess_c = MLP(k, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        # self.preprocess_c = nn.Linear(k, n_hidden)

        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.preprocess(x)
        # c = self.preprocess_c(c)
        x = x + self.placeholder[None, None, :]

        return x




class Postprocess_layer(nn.Module):
    def __init__(self,hidden_dim,out_dim):
        super(Postprocess_layer, self).__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.sigmoid = nn.Sigmoid()
        # self.norm_final = nn.LayerNorm(hidden_dim)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.sigmoid(self.linear(x))
        # x = self.linear(self.norm_final(x))
        return x


class Postprocessing_Model(nn.Module):
    def __init__(self,
                 n_hidden=256,
                 out_dim=1,
                 ref=8,
                 unified_pos=False
                 ):
        super(Postprocessing_Model, self).__init__()
        self.postprocess_layer = Postprocess_layer(n_hidden,out_dim)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # Zero-out output layers:
        nn.init.constant_(self.postprocess_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.postprocess_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.postprocess_layer.linear.weight, 0)
        nn.init.constant_(self.postprocess_layer.linear.bias, 0)
    def forward(self, x, c):
        x = self.postprocess_layer(x, c)
        return x










class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # Input: b, num_points_samples, 3
        # Condition: b, 512
        # Output: b, num_point_samples
        xl = np.linspace(-1,1,32)
        yl = np.linspace(-1,1,32)  
        zl = np.linspace(-1,1,32)
        xc, yc, zc = np.meshgrid(xl, yl, zl, indexing='ij')
        coords = np.concatenate((xc.reshape(-1,1),yc.reshape(-1,1),zc.reshape(-1,1)),axis=1)
        self.coords = torch.tensor(coords, dtype = torch.float32, device = self.device)  
          
        self.preprocessing_model = Preprocessing_Model(n_hidden=256,
                                       input_dim = 3)
        self.attention_model = Attention_Model(n_hidden=256,
                                  n_layers=5,
                                  n_head=8,
                                  mlp_ratio=1,
                                  latent_num=32)

        self.postprocessing_model = Postprocessing_Model(n_hidden=256,
                                  out_dim=1)
        
        self.encoded_feat_linear = nn.Linear(512,256)

    def forward(self, images, args, indices):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        encoded_feat = self.encoded_feat_linear(encoded_feat)
        x = self.coords[indices,:]
        
        x = x.repeat(args.batch_size,1,1)     

        x = self.preprocessing_model(x)
        x = self.attention_model(x, encoded_feat)
        pred = self.postprocessing_model(x, encoded_feat)



        return pred.reshape(args.batch_size,args.num_samples)
