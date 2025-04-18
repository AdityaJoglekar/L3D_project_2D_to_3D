from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
import pytorch3d
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalBatchNorm(nn.Module):
    def __init__(self, latent_size, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.gamma = nn.Linear(latent_size, num_features)
        self.beta = nn.Linear(latent_size, num_features)

    def forward(self, x, latent):
        # x: (B*N, D), latent: (B, D)
        normalized = self.bn(x)
        gamma = self.gamma(latent).unsqueeze(1).expand(-1, x.size(0)//latent.size(0), -1).contiguous().view(-1, self.num_features)
        beta = self.beta(latent).unsqueeze(1).expand(-1, x.size(0)//latent.size(0), -1).contiguous().view(-1, self.num_features)
        return gamma * normalized + beta

class PreActResNetBlock(nn.Module):
    def __init__(self, latent_size, in_dim, out_dim):
        super().__init__()
        self.cbn1 = ConditionalBatchNorm(latent_size, in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.cbn2 = ConditionalBatchNorm(latent_size, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x, latent):
        # Pre-activation structure
        out = F.relu(self.cbn1(x, latent))
        out = self.fc1(out)
        out = F.relu(self.cbn2(out, latent))
        out = self.fc2(out)
        return out + x




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
        xl = np.linspace(-1,1,64)
        yl = np.linspace(-1,1,64)  
        zl = np.linspace(-1,1,64)
        xc, yc, zc = np.meshgrid(xl, yl, zl, indexing='ij')
        coords = np.concatenate((xc.reshape(-1,1),yc.reshape(-1,1),zc.reshape(-1,1)),axis=1)
        self.coords = torch.tensor(coords, dtype = torch.float32, device = self.device)  
        
        self.encoded_feat_linear = nn.Linear(512,256)
        latent_size=256
        self.input_fc = nn.Linear(3, 256)
        self.blocks = nn.ModuleList([
            PreActResNetBlock(latent_size, 256, 256),
            PreActResNetBlock(latent_size, 256, 256),
            PreActResNetBlock(latent_size, 256, 256),
            PreActResNetBlock(latent_size, 256, 256),
            PreActResNetBlock(latent_size, 256, 256),
        ])
        self.final_cbn = ConditionalBatchNorm(latent_size, 256)
        self.final_fc = nn.Linear(256, 1)

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

        B, N = x.shape[:2]
        p_flat = x.view(B*N, 3)

        # Initial projection
        x = self.input_fc(p_flat)  # (B*N, 256)
        
        # ResNet blocks
        for block in self.blocks:
            x = block(x, encoded_feat)
            
        # Final layers
        x = F.relu(self.final_cbn(x, encoded_feat))
        pred = torch.sigmoid(self.final_fc(x))



        return pred.reshape(args.batch_size,args.num_samples)
