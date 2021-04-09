import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils.flow_utils import dump_flow

class ADMMSolverBlock(nn.Module):
    def __init__(self,rho,lamb,eta,grad="1st",T=1):
        super(ADMMSolverBlock, self).__init__()
        # params
        self.T = T
        self.grad = grad
        # variables
        self.beta = None
        self.Q = None
        self.count = 0
        # blocks
        self.get_gradients = Sobel()
        self.apply_threshold = SoftThresholding(rho,lamb)
        self.update_multipliers = MultiplierUpdate(eta)

    def forward(self, F, masks):
        # get masked grads
        dF = self.get_gradients(F) #[dF/dx, dF/dy]
        #dump_flow(F,"F","outflows/{}/".format(self.count))
        #[dump_flow(df,dt,"outflows/{}/".format(self.count)) for dt,df in zip(["dFdx","dFdy"],dF)]
        if self.grad == "2nd":
            dF2 = [self.get_gradients(df) for df in dF] #[[dF/dxx, dF/dxy], [dF/dyx, dF/dyy]]
            dF = [dF2[0][0], dF2[1][1]] #[dF/dxx, dF/dyy]
        c = [df * m for df, m in zip(dF, masks)]
        #[dump_flow(df,dt,"outflows/{}/".format(self.count)) for dt,df in zip(["cdx","cdy"],c)]

        c = torch.cat(c, dim = 1) #[B,4,H,W]
        # initialize 
        beta = torch.zeros_like(c)
        q = torch.zeros_like(c)
        
        Q = [q]
        C = [c]
        Betas = [beta]

        # update q and beta
        for t in range(self.T):
            q = self.apply_threshold(c,beta,t)
            beta = self.update_multipliers(q,c,beta)
            #[dump_flow(q[:,2*i:2*i+2,:,:],dt,"outflows/{}/".format(self.count)) for i, dt in enumerate(["q{}dx".format(t),"q{}dy".format(t)])]

            Q.append(q)
            C.append(c)
            Betas.append(beta)

        #return [Q[-1]], [C[-1]], [Betas[-1]]
        self.count += 1
        return Q, C, Betas
    
class SoftThresholding(nn.Module):
    def __init__(self,rho,lamb):
        super(SoftThresholding, self).__init__()
        if type(lamb) is list: # support several lambda values
            self.lamb = lamb
        else:
            self.lamb = [lamb]
        self.rho = rho
    
    def forward(self,C, beta, i=0):
        th = self.lamb[i] / self.rho

        mask = (C - beta).abs() >= th
        Q = (C - beta - th * torch.sign(C - beta)) * mask
        
        return Q

class MultiplierUpdate(nn.Module):
    def __init__(self, eta):
        super(MultiplierUpdate,self).__init__()
        self.eta = eta

    def forward(self, Q, C, beta):
        beta = beta + self.eta * (Q - C)
        
        return beta

class MaskGenerator(nn.Module):
    def __init__(self,alpha,learn_mask=False):
        super(MaskGenerator,self).__init__()
        self.learn_mask = learn_mask
        self.alpha = alpha
        self.sobel = Sobel()
        if learn_mask:
            self.ddx_encoder = MaskEncoder()
            self.ddy_encoder = MaskEncoder()
    
    def rgb2gray(self, im_rgb):
        #im_rgb = (im_rgb + 1) / 2 # shift (-1,1) to (0,1)
        im_gray = (im_rgb * torch.tensor([[[[0.2989]],[[0.5870]],[[0.1140]]]], dtype = torch.float, device = im_rgb.device)).sum(dim=1,keepdim=True)
        return im_gray

    def forward(self, image, scale=1/8):
        if self.learn_mask: 
            im_grads = self.sobel(self.rgb2gray(image)) #[dx, dy]
            encoders = [self.ddx_encoder, self.ddy_encoder]
            masks = [enc(grad.abs()) for enc, grad in zip(encoders, im_grads)]
        else:
            image = F.interpolate(image, scale_factor=scale, mode='area')
            im_grads = self.sobel(image) #[dx, dy]

            masks = [torch.exp(-torch.mean(torch.abs(grad), 1, keepdim=True) * self.alpha) for grad in im_grads]

        return masks


class MaskEncoder(nn.Module):
    def __init__(self, cin=1):
        super(MaskEncoder,self).__init__()
        self.conv1 = nn.Conv3d(in_channels = cin, out_channels = 8, kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv3d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.out = nn.Conv3d(in_channels = 32, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
        self.relu = nn.ReLU(inplace = True)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))    #spatial dims are //2
        x = self.relu(self.conv2(x))    #spatial dims are //4
        x = self.relu(self.conv3(x))    #spatial dims are //8
        x = self.sig(self.out(x))       #cout = 1, vals are in [0,1]
        
        return x


class Sobel(nn.Module):
    def __init__(self,  f_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        f_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        f_z = []):
        super(Sobel, self).__init__()
        Dx = torch.tensor(f_x, dtype = torch.float, requires_grad = False).view(1,1,3,3)
        Dy = torch.tensor(f_y, dtype = torch.float, requires_grad = False).view(1,1,3,3)
        Dz = torch.tensor(f_z, dtype = torch.float, requires_grad = False).view(1,1,3,3)
        self.D = nn.Parameter(torch.cat((Dx, Dy, Dz), dim = 0), requires_grad = False)
    
    def forward(self, image):
        # apply filter over each channel seperately
        im_ch = torch.split(image, 1, dim = 1)
        grad_ch = [F.conv2d(ch, self.D, padding = 1) for ch in im_ch]
        dx = torch.cat([g[:,0:1,:,:] for g in grad_ch], dim=1)
        dy = torch.cat([g[:,1:2,:,:] for g in grad_ch], dim=1)

        return [dx, dy]
