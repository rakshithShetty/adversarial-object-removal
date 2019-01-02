import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .spectral_normalization import SpectralNorm
from torch.autograd import Variable
from torchvision import models
import sys
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False, only_last = False, final_feat_size=8):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.only_last = only_last
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            if final_feat_size <=64 or type(vgg_pretrained_features[x]) is not nn.MaxPool2d:
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            if final_feat_size <=32 or type(vgg_pretrained_features[x]) is not nn.MaxPool2d:
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            if final_feat_size <=16 or type(vgg_pretrained_features[x]) is not nn.MaxPool2d:
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            if final_feat_size <=8 or type(vgg_pretrained_features[x]) is not nn.MaxPool2d:
                self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        if self.only_last:
            return h_relu5
        else:
            out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
            return out
class AdaptiveScaleTconv(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, scale=2, use_deform=True, n_filters=1):
        super(AdaptiveScaleTconv, self).__init__()
        if int(torch.__version__.split('.')[1])<4:
            self.upsampLayer = nn.Upsample(scale_factor=scale, mode='bilinear')
        else:
            self.upsampLayer = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

        if n_filters > 1:
            self.convFilter = nn.Sequential(*[nn.Conv2d(dim_in if i==0 else dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False) for i in xrange(n_filters)])
        else:
            self.convFilter = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.use_deform = use_deform
        if use_deform:
            self.coordfilter = nn.Conv2d(dim_in, 2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
            self.coordfilter.weight.data.zero_()
        #Identity transform used to create a regular grid!

    def forward(self, x, extra_inp=None):
        # First upsample the input with transposed/ upsampling
        # Compute the warp co-ordinates using a conv
        # Warp the conv
        up_out = self.upsampLayer(x)
        filt_out = self.convFilter(up_out if extra_inp is None else torch.cat([up_out,extra_inp], dim=1))

        if self.use_deform:
            cord_offset = self.coordfilter(up_out)
            reg_grid = Variable(torch.FloatTensor(np.stack(np.meshgrid(np.linspace(-1,1, up_out.size(2)), np.linspace(-1,1, up_out.size(3))))).cuda(),requires_grad=False)
            deform_grid = reg_grid.detach() + F.tanh(cord_offset)
            deformed_out = F.grid_sample(filt_out, deform_grid.transpose(1,3).transpose(1,2), mode='bilinear', padding_mode='zeros')
            feat_out = (deform_grid, reg_grid, cord_offset)
        else:
            deformed_out = filt_out
            feat_out = []

        #Deformed out
        return deformed_out, feat_out


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dilation=1, padtype = 'zero'):
        super(ResidualBlock, self).__init__()
        pad = dilation
        layers = []
        if padtype== 'reflection':
            layers.append(nn.ReflectionPad2d(pad)); pad=0
        elif padtype == 'replication':
            layers.append(nn.ReplicationPad2d(p)); pad=0

        layers.extend([ nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=False),
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.1,inplace=True)])

        pad = dilation
        if padtype== 'reflection':
            layers.append(nn.ReflectionPad2d(pad)); pad=0
        elif padtype == 'replication':
            layers.append(nn.ReplicationPad2d(p)); pad=0

        layers.extend([
            nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=False),
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.1,inplace=True)
            ])

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.main(x)

class ResidualBlockBnorm(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dilation=1, padtype = 'zero'):
        super(ResidualBlockBnorm, self).__init__()
        pad = dilation
        layers = []
        if padtype == 'reflection':
            layers.append(nn.ReflectionPad2d(pad)); pad=0
        elif padtype == 'replication':
            layers.append(nn.ReplicationPad2d(p)); pad=0

        layers.extend([ nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.1,inplace=True)])

        pad = dilation
        if padtype== 'reflection':
            layers.append(nn.ReflectionPad2d(pad)); pad=0
        elif padtype == 'replication':
            layers.append(nn.ReplicationPad2d(p)); pad=0

        layers.extend([
            nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.1,inplace=True)
            ])

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.main(x)

class ResidualBlockNoNorm(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dilation=1, padtype = 'zero'):
        super(ResidualBlockNoNorm, self).__init__()
        pad = dilation
        layers = []
        if padtype == 'reflection':
            layers.append(nn.ReflectionPad2d(pad)); pad=0
        elif padtype == 'replication':
            layers.append(nn.ReplicationPad2d(p)); pad=0

        layers.extend([ nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=False),
            nn.LeakyReLU(0.1,inplace=True)])

        pad = dilation
        if padtype== 'reflection':
            layers.append(nn.ReflectionPad2d(pad)); pad=0
        elif padtype == 'replication':
            layers.append(nn.ReplicationPad2d(p)); pad=0

        layers.extend([
            nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=False),
            nn.LeakyReLU(0.1,inplace=True)
            ])

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.main(x)
class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, g_smooth_layers=0, binary_mask=0):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class GeneratorDiff(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, g_smooth_layers=0, binary_mask=0):
        super(GeneratorDiff, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        # Remove this non-linearity or use 2.0*tanh ?
        layers.append(nn.Tanh())
        self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1)
        self.main = nn.Sequential(*layers)

    def forward(self, x, c, out_diff = False):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        xcat = torch.cat([x, c], dim=1)
        net_out = self.main(xcat)
        if out_diff:
            return (x+2.0*net_out), net_out
        else:
            return (x+2.0*net_out)

class GeneratorDiffWithInp(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=3, g_smooth_layers=0, binary_mask=0):
        super(GeneratorDiffWithInp, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim))

        # Up-Sampling
        self.up_sampling_convlayers = nn.ModuleList()
        self.up_sampling_inorm= nn.ModuleList()
        self.up_sampling_ReLU= nn.ModuleList()
        for i in range(2):
            self.up_sampling_convlayers.append(nn.ConvTranspose2d(curr_dim+3, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            self.up_sampling_inorm.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            self.up_sampling_ReLU.append(nn.ReLU(inplace=False))
            curr_dim = curr_dim // 2

        self.final_Layer = nn.Conv2d(curr_dim+3, 3, kernel_size=7, stride=1, padding=3, bias=False)
        #layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        # Remove this non-linearity or use 2.0*tanh ?
        self.finalNonLin = nn.Tanh()

        self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1)
        self.main = nn.Sequential(*layers)

    def forward(self, x, c, out_diff = False):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        xcat = torch.cat([x, c], dim=1)
        bottle_out = self.main(xcat)
        curr_downscale = 4
        up_inp = [torch.cat([bottle_out,nn.functional.avg_pool2d(x,curr_downscale)], dim=1)]
        curr_downscale = curr_downscale//2
        up_out = []
        for i in range(len(self.up_sampling_convlayers)):
            #self.up_sampling_convlayers(x
            up_out.append(self.up_sampling_ReLU[i](self.up_sampling_inorm[i](self.up_sampling_convlayers[i](up_inp[i]))))
            up_inp.append(torch.cat([up_out[i],nn.functional.avg_pool2d(x,curr_downscale)], dim=1))
            curr_downscale = curr_downscale//2

        net_out = self.finalNonLin(self.final_Layer(up_inp[-1]))

        if out_diff:
            return (x+2.0*net_out), net_out
        else:
            return (x+2.0*net_out)

class GeneratorDiffAndMask(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=3, g_smooth_layers=0, binary_mask=0):
        super(GeneratorDiffAndMask, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim))

        # Up-Sampling Differential layers
        self.up_sampling_convlayers = nn.ModuleList()
        self.up_sampling_inorm= nn.ModuleList()
        self.up_sampling_ReLU= nn.ModuleList()

        # Up-Sampling Mask layers
        self.up_sampling_convlayers_mask = nn.ModuleList()
        self.up_sampling_inorm_mask= nn.ModuleList()
        self.up_sampling_ReLU_mask = nn.ModuleList()
        for i in range(2):
            self.up_sampling_convlayers.append(nn.ConvTranspose2d(curr_dim+3, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            self.up_sampling_inorm.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            self.up_sampling_ReLU.append(nn.ReLU(inplace=False))

            ## Add the mask path
            self.up_sampling_convlayers_mask.append(nn.ConvTranspose2d(curr_dim+3, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            self.up_sampling_inorm_mask.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            self.up_sampling_ReLU_mask.append(nn.ReLU(inplace=False))
            curr_dim = curr_dim // 2

        self.final_Layer = nn.Conv2d(curr_dim+3, 3, kernel_size=7, stride=1, padding=3, bias=False)
        #layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        # Remove this non-linearity or use 2.0*tanh ?
        self.finalNonLin = nn.Tanh()

        self.final_Layer_mask = nn.Conv2d(curr_dim+3, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.finalNonLin_mask = nn.Sigmoid()


        self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1)
        self.main = nn.Sequential(*layers)

    def forward(self, x, c, out_diff = False):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        xcat = torch.cat([x, c], dim=1)
        bottle_out = self.main(xcat)
        curr_downscale = 4
        up_inp = [torch.cat([bottle_out,nn.functional.avg_pool2d(x,curr_downscale)], dim=1)]
        up_inp_mask = [None]
        up_inp_mask[0] = up_inp[0]
        curr_downscale = curr_downscale//2
        up_out = []
        up_out_mask = []
        for i in range(len(self.up_sampling_convlayers)):
            #self.up_sampling_convlayers(x
            up_out.append(self.up_sampling_ReLU[i](self.up_sampling_inorm[i](self.up_sampling_convlayers[i](up_inp[i]))))
            up_inp.append(torch.cat([up_out[i],nn.functional.avg_pool2d(x,curr_downscale)], dim=1))

            # Compute the maks output
            up_out_mask.append(self.up_sampling_ReLU_mask[i](self.up_sampling_inorm_mask[i](self.up_sampling_convlayers_mask[i](up_inp_mask[i]))))
            up_inp_mask.append(torch.cat([up_out_mask[i],nn.functional.avg_pool2d(x,curr_downscale)], dim=1))
            curr_downscale = curr_downscale//2

        net_out = self.finalNonLin(self.final_Layer(up_inp[-1]))
        mask = self.finalNonLin_mask(self.final_Layer_mask(up_inp_mask[-1]))
        if out_diff:
            return ((1-mask)*x+mask*net_out), (net_out, mask)
        else:
            return ((1-mask)*x+mask*net_out)

class GeneratorDiffAndMask_V2(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=3, g_smooth_layers=0, binary_mask=0):
        super(GeneratorDiffAndMask_V2, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim))

        # Up-Sampling Differential layers
        self.up_sampling_convlayers = nn.ModuleList()
        self.up_sampling_inorm= nn.ModuleList()
        self.up_sampling_ReLU= nn.ModuleList()

        for i in range(2):
            self.up_sampling_convlayers.append(nn.ConvTranspose2d(curr_dim+3, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            self.up_sampling_inorm.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            self.up_sampling_ReLU.append(nn.ReLU(inplace=False))

            curr_dim = curr_dim // 2

        self.final_Layer = nn.Conv2d(curr_dim+3, 3, kernel_size=7, stride=1, padding=3, bias=False)
        #layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        # Remove this non-linearity or use 2.0*tanh ?
        self.finalNonLin = nn.Tanh()

        self.final_Layer_mask = nn.Conv2d(curr_dim+3, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.finalNonLin_mask = nn.Sigmoid()

        self.g_smooth_layers = g_smooth_layers
        if g_smooth_layers > 0:
            smooth_layers = []
            for i in range(g_smooth_layers):
                smooth_layers.append(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False))
                smooth_layers.append(nn.Tanh())
            self.smooth_layers= nn.Sequential(*smooth_layers)

        self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1)
        self.binary_mask = binary_mask
        self.main = nn.Sequential(*layers)

    def forward(self, x, c, out_diff = False):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        xcat = torch.cat([x, c], dim=1)
        bottle_out = self.main(xcat)
        curr_downscale = 4
        up_inp = [torch.cat([bottle_out,nn.functional.avg_pool2d(x,curr_downscale)], dim=1)]
        curr_downscale = curr_downscale//2
        up_out = []
        up_out_mask = []
        for i in range(len(self.up_sampling_convlayers)):
            #self.up_sampling_convlayers(x
            up_out.append(self.up_sampling_ReLU[i](self.up_sampling_inorm[i](self.up_sampling_convlayers[i](up_inp[i]))))
            up_inp.append(torch.cat([up_out[i],nn.functional.avg_pool2d(x,curr_downscale)], dim=1))

            curr_downscale = curr_downscale//2

        net_out = self.finalNonLin(self.final_Layer(up_inp[-1]))
        mask = self.finalNonLin_mask(2.0*self.final_Layer_mask(up_inp[-1]))

        if self.binary_mask:
            mask = ((mask>0.5).float()- mask).detach() + mask

        masked_image = ((1-mask)*x+(mask)*(2.0*net_out))

        if self.g_smooth_layers > 0:
            out_image = self.smooth_layers(masked_image)
        else:
            out_image = masked_image

        if out_diff:
            return out_image, (net_out, mask)
        else:
            return out_image

def get_conv_inorm_relu_block(i, o, k, s, p, slope=0.1, padtype='zero', dilation=1):
    layers = []
    if padtype == 'reflection':
        layers.append(nn.ReflectionPad2d(p)); p=0
    elif padtype == 'replication':
        layers.append(nn.ReplicationPad2d(p)); p=0
    layers.append(nn.Conv2d(i, o, kernel_size=k, stride=s, padding=p, dilation=dilation, bias=False))
    layers.append(nn.InstanceNorm2d(o, affine=True))
    layers.append(nn.LeakyReLU(slope,inplace=True))
    return layers

class GeneratorOnlyMask(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=5, g_smooth_layers=0, binary_mask=0):
        super(GeneratorOnlyMask, self).__init__()

        layers = []
        layers.extend(get_conv_inorm_relu_block(3+c_dim, conv_dim, 7, 1, 3, padtype='zero'))

        # Down-Sampling
        curr_dim = conv_dim

        for i in range(3):
            layers.extend(get_conv_inorm_relu_block(curr_dim, curr_dim*2, 4, 2, 1, padtype='zero'))
            curr_dim = curr_dim * 2

        dilation=1
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dilation=dilation, padtype='zero'))
            if i> 1:
                # This gives dilation as 1, 1, 2, 4, 8, 16
                dilation=dilation*2

        # Up-Sampling Differential layers
        self.up_sampling_convlayers = nn.ModuleList()
        self.up_sampling_inorm= nn.ModuleList()
        self.up_sampling_ReLU= nn.ModuleList()

        for i in range(3):
            self.up_sampling_convlayers.append(nn.ConvTranspose2d(curr_dim+3, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            self.up_sampling_inorm.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            self.up_sampling_ReLU.append(nn.ReLU(inplace=False))

            curr_dim = curr_dim // 2

        self.final_Layer_mask = nn.Conv2d(curr_dim+3, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.finalNonLin_mask = nn.Sigmoid()

        self.g_smooth_layers = g_smooth_layers
        if g_smooth_layers > 0:
            smooth_layers = []
            for i in range(g_smooth_layers):
                smooth_layers.append(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False))
                smooth_layers.append(nn.Tanh())
            self.smooth_layers= nn.Sequential(*smooth_layers)

        self.binary_mask = binary_mask
        self.main = nn.Sequential(*layers)

    def forward(self, x, c, out_diff = False):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        xcat = torch.cat([x, c], dim=1)
        bottle_out = self.main(xcat)
        curr_downscale = 8
        up_inp = [torch.cat([bottle_out,nn.functional.avg_pool2d(x,curr_downscale)], dim=1)]
        curr_downscale = curr_downscale//2
        up_out = []
        up_out_mask = []
        for i in range(len(self.up_sampling_convlayers)):
            #self.up_sampling_convlayers(x
            up_out.append(self.up_sampling_ReLU[i](self.up_sampling_inorm[i](self.up_sampling_convlayers[i](up_inp[i]))))
            up_inp.append(torch.cat([up_out[i],nn.functional.avg_pool2d(x,curr_downscale)], dim=1))

            curr_downscale = curr_downscale//2

        mask = self.finalNonLin_mask(2.0*self.final_Layer_mask(up_inp[-1]))

        if self.binary_mask:
            mask = ((mask>0.5).float()- mask).detach() + mask

        masked_image = (1-mask)*x #+(mask)*(2.0*net_out))

        out_image = masked_image

        if out_diff:
            return out_image, mask
        else:
            return out_image

class GeneratorMaskAndFeat(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=5, g_smooth_layers=0, binary_mask=0, out_feat_dim=256, up_sampling_type='bilinear',
                 n_upsamp_filt=2, mask_size = 0, additional_cond='image', per_classMask=0, noInpLabel=0, mask_normalize = False, nc=3,
                 use_bias = False, use_bnorm = 0, cond_inp_pnet=0, cond_parallel_track = 0):
        super(GeneratorMaskAndFeat, self).__init__()

        self.lowres_mask = int(mask_size <= 32)
        self.per_classMask = per_classMask
        self.additional_cond = additional_cond
        self.noInpLabel = noInpLabel
        self.mask_normalize = mask_normalize
        layers = []
        # Image is 128 x 128
        layers.extend(get_conv_inorm_relu_block(nc if noInpLabel else nc+c_dim, conv_dim, 7, 1, 3, padtype='zero'))

        # Down-Sampling
        curr_dim = conv_dim
        extra_dim = 3 if self.additional_cond == 'image' else c_dim if self.additional_cond == 'label'else 0

        #-------------------------------------------
        # After downsampling spatial dim is 16 x 16
        # Feat dim is 512
        #-------------------------------------------
        for i in range(3 - self.lowres_mask):
            layers.extend(get_conv_inorm_relu_block(curr_dim, curr_dim*2, 4, 2, 1, padtype='zero'))
            curr_dim = curr_dim * 2

        dilation=1
        #-------------------------------------------
        # After residual spatial dim is 16 x 16
        # Feat dim is 512
        #-------------------------------------------
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dilation=dilation, padtype='zero'))
            if i> 1:
                # This gives dilation as 1, 1, 2, 4, 8, 16
                dilation=dilation*2

        # Up-Sampling Differential layers
        self.up_sampling_convlayers = nn.ModuleList()
        if self.lowres_mask == 0:
            self.up_sampling_inorm= nn.ModuleList()
            self.up_sampling_ReLU= nn.ModuleList()

        self.out_feat_dim = out_feat_dim
        if out_feat_dim > 0:
            featGenLayers = []
            #-------------------------------------------
            # After featGen layers spatial dim is 1 x 1
            # Feat dim is 512
            #-------------------------------------------
            for i in xrange(3):
                featGenLayers.extend(get_conv_inorm_relu_block(curr_dim, curr_dim, 3, 1, 1, padtype='zero'))
                featGenLayers.append(nn.MaxPool2d(2) if i<2 else nn.MaxPool2d(4))

            self.featGenConv = nn.Sequential(*featGenLayers)
            self.featGenLin = nn.Linear(curr_dim, out_feat_dim)

        for i in range(3-self.lowres_mask):
            if self.lowres_mask == 0:
                if up_sampling_type== 't_conv':
                    self.up_sampling_convlayers.append(nn.ConvTranspose2d(curr_dim+extra_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=use_bias))
                elif up_sampling_type == 'nearest':
                    self.up_sampling_convlayers.append(nn.Upsample(scale_factor=2, mode='nearest'))
                    self.up_sampling_convlayers.append(nn.Conv2d(curr_dim+extra_dim, curr_dim//2, kernel_size=3, stride=1, padding=1, bias=use_bias))
                elif up_sampling_type == 'deform':
                    self.up_sampling_convlayers.append(AdaptiveScaleTconv(curr_dim+extra_dim, curr_dim//2, scale=2, n_filters=n_upsamp_filt))
                elif up_sampling_type == 'bilinear':
                    self.up_sampling_convlayers.append(AdaptiveScaleTconv(curr_dim+extra_dim, curr_dim//2, scale=2, use_deform=False, n_filters=n_upsamp_filt))
                self.up_sampling_inorm.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
                self.up_sampling_ReLU.append(nn.ReLU(inplace=False))
            else:
                # In this case just use more residual blocks to drop dimensions
                self.up_sampling_convlayers.append(nn.Sequential(*get_conv_inorm_relu_block(curr_dim+extra_dim, curr_dim//2, 3, 1, 1, padtype='zero')))


            curr_dim = curr_dim // 2

        self.final_Layer_mask = nn.Conv2d(curr_dim+extra_dim, c_dim+1 if per_classMask else 1, kernel_size=7, stride=1, padding=3, bias=True if mask_normalize else use_bias)
        self.finalNonLin_mask = nn.Sigmoid()

        self.g_smooth_layers = g_smooth_layers
        if g_smooth_layers > 0:
            smooth_layers = []
            for i in range(g_smooth_layers):
                smooth_layers.append(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False))
                smooth_layers.append(nn.Tanh())
            self.smooth_layers= nn.Sequential(*smooth_layers)

        self.binary_mask = binary_mask
        self.main = nn.Sequential(*layers)

    def prepInp(self, feat, x, c, curr_scale):
        if self.additional_cond == 'image':
            up_inp = torch.cat([feat,nn.functional.avg_pool2d(x,curr_scale)], dim=1)
        elif self.additional_cond == 'label':
            up_inp = torch.cat([feat,nn.functional.avg_pool2d(c,curr_scale)], dim=1)
        else:
            up_inp = feat
        return up_inp

    def forward(self, x, c, out_diff = False, binary_mask=False, mask_threshold = 0.3):
        # replicate spatially and concatenate domain information
        bsz = x.size(0)
        if self.per_classMask:
            maxC,cIdx = c.max(dim=1)
            cIdx[maxC==0] = c.size(1) + 1 if self.mask_normalize else c.size(1)

        if self.noInpLabel:
            xcat = x
        else:
            c = c.unsqueeze(2).unsqueeze(3)
            c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
            xcat = torch.cat([x, c], dim=1)
        bottle_out = self.main(xcat)
        curr_downscale = 8 if self.lowres_mask == 0 else 4
        up_inp = [self.prepInp(bottle_out, x, c, curr_downscale)]
        curr_downscale = curr_downscale//2 if self.lowres_mask == 0 else curr_downscale
        up_out = []
        for i in range(len(self.up_sampling_convlayers)):
            #self.up_sampling_convlayers(x
            if type(self.up_sampling_convlayers[i]) == AdaptiveScaleTconv:
                upsampout,_ = self.up_sampling_convlayers[i](up_inp[i])
            else:
                upsampout = self.up_sampling_convlayers[i](up_inp[i])
            up_out.append(self.up_sampling_ReLU[i](self.up_sampling_inorm[i](upsampout)) if self.lowres_mask == 0 else upsampout)
            up_inp.append(self.prepInp(up_out[i], x, c, curr_downscale))

            curr_downscale = curr_downscale//2 if self.lowres_mask == 0 else curr_downscale

        allmasks = self.final_Layer_mask(up_inp[-1])
        if self.mask_normalize:
            allmasks = torch.cat([F.softmax(allmasks, dim=1), torch.zeros_like(allmasks[:,0:1,::]).detach()], dim=1)
        chosenMask = allmasks if (self.per_classMask==0) else allmasks[torch.arange(cIdx.size(0)).long().cuda(),cIdx,::].view(bsz,1,allmasks.size(2), allmasks.size(3))
        if not self.mask_normalize:
            mask = self.finalNonLin_mask(2.0*chosenMask)
        else:
            mask = chosenMask

        if self.out_feat_dim > 0:
            out_feat = self.featGenLin(self.featGenConv(bottle_out).view(bsz,-1))
        else:
            out_feat = None

        if self.binary_mask or binary_mask:
            if self.mask_normalize:
                maxV,_ = allmasks.max(dim=1)
                mask = (torch.ge(mask, maxV.view(mask.size())).float()- mask).detach() + mask
            else:
                mask = ((mask>=mask_threshold).float()- mask).detach() + mask


        #masked_image = (1-mask)*x #+(mask)*(2.0*net_out))

        #out_image = masked_image

        return None, mask, out_feat, allmasks

class GeneratorMaskAndFeat_ImNetBackbone(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=5, g_smooth_layers=0, binary_mask=0, out_feat_dim=256, up_sampling_type='bilinear',
                 n_upsamp_filt=2, mask_size = 0, additional_cond='image', per_classMask=0, noInpLabel=0, mask_normalize = False, nc=3, use_bias = False,
                 net_type='vgg19', use_bnorm = 0, cond_inp_pnet=0):
        super(GeneratorMaskAndFeat_ImNetBackbone, self).__init__()

        self.pnet = Vgg19() if net_type == 'vgg19' else None
        self.per_classMask = per_classMask
        self.additional_cond = additional_cond
        self.noInpLabel = noInpLabel
        self.mask_normalize = mask_normalize
        self.nc = nc
        self.out_feat_dim = out_feat_dim
        self.binary_mask = binary_mask
        # Down-Sampling
        curr_dim = conv_dim
        extra_dim = 3 if self.additional_cond == 'image' else c_dim if self.additional_cond == 'label'else 0
        layers = nn.ModuleList()
        if nc > 3:
            extra_dim = extra_dim + 1
            self.appendGtInp = True
        else:
            self.appendGtInp = False

        ResBlock  = ResidualBlockBnorm if use_bnorm==1 else ResidualBlock if use_bnorm==2 else ResidualBlockNoNorm
        #===========================================================
        # Three blocks of layers:
        # Feature absorb layer --> Residual block --> Upsampling
        #===========================================================
        # First block This takes input features of 512x8x8 dims
        # Upsample to 16x16
        layers.append(nn.Conv2d(512+extra_dim, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(ResBlock(dim_in=512,dilation=1, padtype='zero'))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        #-----------------------------------------------------------
        # Second Block - This takes input features of 512x16x16 from Layer 1 and 512x16x16 from VGG
        # Upsample to 32x32
        layers.append(nn.Conv2d(1024+extra_dim, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(ResBlock(dim_in=512,dilation=1, padtype='zero'))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        #-----------------------------------------------------------
        # Third layer
        # This takes input features of 256x32x32 from Layer 1 and 256x32x32 from VGG
        layers.append(nn.Conv2d(512+256+extra_dim, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(ResBlock(dim_in=512,dilation=1, padtype='zero'))

        self.layers = layers

        self.final_Layer_mask = nn.Conv2d(512+extra_dim, c_dim+1 if per_classMask else 1, kernel_size=7, stride=1, padding=3, bias=True if mask_normalize else use_bias)
        self.finalNonLin_mask = nn.Sigmoid()

        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1), requires_grad=False).cuda()
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1), requires_grad=False).cuda()

    def prepInp(self, feat, img, c, gtmask):
        if self.additional_cond == 'image':
            up_inp = torch.cat([feat,nn.functional.adaptive_avg_pool2d(img,feat.size(-1))], dim=1)
        elif self.additional_cond == 'label':
            up_inp = torch.cat([feat,c.expand(c.size(0), c.size(1), feat.size(2), feat.size(3))], dim=1)
        else:
            up_inp = feat
        if self.appendGtInp:
            up_inp = torch.cat([up_inp, nn.functional.adaptive_max_pool2d(gtmask,up_inp.size(-1))], dim=1)
        return up_inp

    def forward(self, x, c, out_diff = False, binary_mask=False, mask_threshold = 0.3):
        # replicate spatially and concatenate domain information
        bsz = x.size(0)
        img = x[:,:3,::]
        gtmask = x[:,3:,::]  if self.appendGtInp else None
        if self.per_classMask:
            maxC,cIdx = c.max(dim=1)
            cIdx[maxC==0] = c.size(1) + 1 if self.mask_normalize else c.size(1)

        c = c.unsqueeze(2).unsqueeze(3)
        img = (img - self.shift.expand_as(img))/self.scale.expand_as(img)

        vgg_out = self.pnet(img)
        up_inp = [self.prepInp(vgg_out[-1], img, c, gtmask)]
        for i in range(len(self.layers)):
            #self.up_sampling_convlayers(img
            upsampout = self.layers[i](up_inp[-1])
            up_inp.append(upsampout)
            if i%4 == 3:
                up_inp.append(self.prepInp(torch.cat([up_inp[-1],vgg_out[-1-(i+1)//4]],dim=1), img, c, gtmask))
        up_inp.append(self.prepInp(up_inp[-1], img, c, gtmask))

        allmasks = self.final_Layer_mask(up_inp[-1])
        if self.mask_normalize:
            allmasks = torch.cat([F.softmax(allmasks, dim=1), torch.zeros_like(allmasks[:,0:1,::]).detach()], dim=1)
        chosenMask = allmasks if (self.per_classMask==0) else allmasks[torch.arange(cIdx.size(0)).long().cuda(),cIdx,::].view(bsz,1,allmasks.size(2), allmasks.size(3))
        if not self.mask_normalize:
            mask = self.finalNonLin_mask(2.0*chosenMask)
        else:
            mask = chosenMask

        if self.out_feat_dim > 0:
            out_feat = self.featGenLin(self.featGenConv(bottle_out).view(bsz,-1))
        else:
            out_feat = None

        if self.binary_mask or binary_mask:
            if self.mask_normalize:
                maxV,_ = allmasks.max(dim=1)
                mask = (torch.ge(mask, maxV.view(mask.size())).float()- mask).detach() + mask
            else:
                mask = ((mask>=mask_threshold).float()- mask).detach() + mask


        #masked_image = (1-mask)*x #+(mask)*(2.0*net_out))

        #out_image = masked_image

        return None, mask, out_feat, allmasks

class GeneratorMaskAndFeat_ImNetBackbone_V2(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=5, g_smooth_layers=0, binary_mask=0, out_feat_dim=256, up_sampling_type='bilinear',
                 n_upsamp_filt=2, mask_size = 0, additional_cond='image', per_classMask=0, noInpLabel=0, mask_normalize = False, nc=3, use_bias = False,
                 net_type='vgg19', use_bnorm = 0, cond_inp_pnet=0, cond_parallel_track= 0):
        super(GeneratorMaskAndFeat_ImNetBackbone_V2, self).__init__()

        self.pnet = Vgg19(final_feat_size=mask_size) if net_type == 'vgg19' else None
        self.mask_size = mask_size
        self.per_classMask = per_classMask
        self.additional_cond = additional_cond
        self.noInpLabel = noInpLabel
        self.mask_normalize = mask_normalize
        self.nc = nc
        self.out_feat_dim = out_feat_dim
        self.cond_inp_pnet = cond_inp_pnet
        self.cond_parallel_track = cond_parallel_track
        # Down-Sampling
        curr_dim = conv_dim
        extra_dim = 3 if self.additional_cond == 'image' else c_dim if self.additional_cond == 'label'else 0
        layers = nn.ModuleList()
        if nc > 3:
            extra_dim = extra_dim# + 1
            self.appendGtInp = False #True
        else:
            self.appendGtInp = False

        ResBlock  = ResidualBlockBnorm if use_bnorm==1 else ResidualBlock if use_bnorm==2 else ResidualBlockNoNorm
        #===========================================================
        # Three blocks of layers:
        # Feature absorb layer --> Residual block --> Upsampling
        #===========================================================
        # First block This takes input features of 512x32x32 dims
        # Upsample to 16x16
        start_dim = 512
        gt_cond_dim =  0 if cond_inp_pnet else int(nc>3)*self.cond_parallel_track if self.cond_parallel_track else int(nc>3)
        if self.cond_parallel_track:
            cond_parallel_layers = [] #nn.ModuleList()
            cond_parallel_layers.append(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1))
            cond_parallel_layers.append(nn.LeakyReLU(0.1, inplace=True))
            cond_parallel_layers.append(nn.Conv2d(64, self.cond_parallel_track, kernel_size=3, stride=1, padding=1))
            cond_parallel_layers.append(nn.LeakyReLU(0.1, inplace=True))
            self.cond_parallel_layers = nn.Sequential(*cond_parallel_layers)

        layers.append(nn.Conv2d(512+extra_dim + gt_cond_dim, start_dim, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(ResBlock(dim_in=start_dim,dilation=1, padtype='zero'))
        #layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        #-----------------------------------------------------------
        # Second Block - This takes input features of 512x16x16 from Layer 1 and 512x16x16 from VGG
        # Upsample to 32x32
        layers.append(nn.Conv2d(start_dim+extra_dim, start_dim//2, kernel_size=3, stride=1, padding=1))
        start_dim = start_dim // 2
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(ResBlock(dim_in=start_dim,dilation=1, padtype='zero'))
        #-----------------------------------------------------------
        # Third layer
        # This takes input features of 256x32x32 from Layer 1 and 256x32x32 from VGG
        layers.append(nn.Conv2d(start_dim+extra_dim, start_dim//2, kernel_size=3, stride=1, padding=1))
        start_dim = start_dim // 2
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(ResBlock(dim_in=start_dim,dilation=1, padtype='zero'))

        self.layers = layers

        self.final_Layer_mask = nn.Conv2d(start_dim+extra_dim, c_dim+1 if per_classMask else 1, kernel_size=7, stride=1, padding=3, bias=True if mask_normalize else use_bias)
        self.finalNonLin_mask = nn.Sigmoid()

        self.binary_mask = binary_mask
        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1), requires_grad=False).cuda()
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1), requires_grad=False).cuda()

    def prepInp(self, feat, img, c, gtmask):
        if self.additional_cond == 'image':
            up_inp = torch.cat([feat,nn.functional.adaptive_avg_pool2d(img,feat.size(-1))], dim=1)
        elif self.additional_cond == 'label':
            up_inp = torch.cat([feat,c.expand(c.size(0), c.size(1), feat.size(2), feat.size(3))], dim=1)
        else:
            up_inp = feat
        if self.appendGtInp:
            up_inp = torch.cat([up_inp, nn.functional.adaptive_max_pool2d(gtmask,up_inp.size(-1))], dim=1)
        return up_inp

    def forward(self, x, c, out_diff = False, binary_mask=False, mask_threshold = 0.3):
        # replicate spatially and concatenate domain information
        bsz = x.size(0)
        gtmask = x[:,3:,::] if x.size(1) > 3 else None
        img = x[:,:3,::]
        img = (img - self.shift.expand_as(img))/self.scale.expand_as(img)
        if self.cond_inp_pnet:
            img = img*gtmask
        if self.cond_parallel_track and gtmask is not None:
            gtfeat = self.cond_parallel_layers(gtmask)
        else:
            gtfeat = gtmask

        if self.per_classMask:
            maxC,cIdx = c.max(dim=1)
            cIdx[maxC==0] = c.size(1) + 1 if self.mask_normalize else c.size(1)

        c = c.unsqueeze(2).unsqueeze(3)

        vgg_out = self.pnet(img)
        #up_inp = [self.prepInp(vgg_out[-1], img, c, gtmask)]
        if (gtfeat is not None) and (not self.cond_inp_pnet):
            up_inp = [torch.cat([self.prepInp(vgg_out[-1], img, c, gtfeat), nn.functional.adaptive_max_pool2d(gtfeat,vgg_out[-1].size(-1))],dim=1)]
        else:
            up_inp = [self.prepInp(vgg_out[-1], img, c, gtfeat)]

        for i in range(len(self.layers)):
            #self.up_sampling_convlayers(img
            upsampout = self.layers[i](up_inp[-1])
            up_inp.append(upsampout)
            if i%3 == 2:
                up_inp.append(self.prepInp(upsampout, img, c, gtfeat))

        allmasks = self.final_Layer_mask(up_inp[-1])
        if self.mask_normalize:
            allmasks = torch.cat([F.softmax(allmasks, dim=1), torch.zeros_like(allmasks[:,0:1,::]).detach()], dim=1)
        chosenMask = allmasks if (self.per_classMask==0) else allmasks[torch.arange(cIdx.size(0)).long().cuda(),cIdx,::].view(bsz,1,allmasks.size(2), allmasks.size(3))
        if not self.mask_normalize:
            mask = self.finalNonLin_mask(2.0*chosenMask)
        else:
            mask = chosenMask

        if self.out_feat_dim > 0:
            out_feat = self.featGenLin(self.featGenConv(bottle_out).view(bsz,-1))
        else:
            out_feat = None

        if self.binary_mask or binary_mask:
            if self.mask_normalize:
                maxV,_ = allmasks.max(dim=1)
                mask = (torch.ge(mask, maxV.view(mask.size())).float()- mask).detach() + mask
            else:
                mask = ((mask>=mask_threshold).float()- mask).detach() + mask


        #masked_image = (1-mask)*x #+(mask)*(2.0*net_out))

        #out_image = masked_image

        return None, mask, out_feat, allmasks


class GeneratorBoxReconst(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, feat_dim=128, repeat_num=6, g_downsamp_layers=2, dil_start =0,
                 up_sampling_type='t_conv', padtype='zero', nc=3, n_upsamp_filt=1, gen_full_image=0):
        super(GeneratorBoxReconst, self).__init__()

        downsamp_layers = []
        layers = []
        downsamp_layers.extend(get_conv_inorm_relu_block(nc+1, conv_dim, 7, 1, 3, padtype=padtype))
        self.g_downsamp_layers = g_downsamp_layers
        self.gen_full_image = gen_full_image

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(g_downsamp_layers):
            downsamp_layers.extend(get_conv_inorm_relu_block(curr_dim, curr_dim*2, 4, 2, 1, padtype=padtype))
            curr_dim = curr_dim * 2

        # Bottleneck
        # Here- input the target features
        dilation=1
        if feat_dim > 0:
            layers.extend(get_conv_inorm_relu_block(curr_dim+feat_dim, curr_dim, 3, 1, 1, padtype=padtype, dilation=dilation))
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dilation=dilation, padtype=padtype))
            if i> dil_start:
                # This gives dilation as 1, 1, 2, 4, 8, 16
                dilation=dilation*2

        # Up-Sampling
        for i in range(g_downsamp_layers):
            if up_sampling_type== 't_conv':
                layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            elif up_sampling_type == 'nearest':
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
                layers.append(nn.Conv2d(curr_dim, curr_dim//2, kernel_size=3, stride=1, padding=1, bias=False))
            elif up_sampling_type == 'deform':
                layers.append(AdaptiveScaleTconv(curr_dim+(self.gen_full_image * curr_dim//2), curr_dim//2, scale=2, n_filters=n_upsamp_filt))
            elif up_sampling_type == 'bilinear':
                layers.append(AdaptiveScaleTconv(curr_dim+(self.gen_full_image * curr_dim//2), curr_dim//2, scale=2, use_deform=False))

            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.LeakyReLU(0.1,inplace=True))
            curr_dim = curr_dim // 2

        pad=3
        if padtype=='reflection':
            layers.append(nn.ReflectionPad2d(pad)); pad=0
        layers.append(nn.Conv2d(curr_dim, nc, kernel_size=7, stride=1, padding=pad, bias=False))
        # Remove this non-linearity or use 2.0*tanh ?
        layers.append(nn.Tanh())
        self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1)
        self.downsample = nn.Sequential(*downsamp_layers)
        #self.generate = nn.Sequential(*layers)
        self.generate = nn.ModuleList(layers)

    def forward(self, x, feat, out_diff = False):
        w, h = x.size(2), x.size(3)
        # This is just to makes sure that when we pass it through the downsampler we don't lose some width and height
        xI = F.pad(x,(0,(8-h%8)%8,0,(8 - w%8)%8),mode='replicate')
        #print(xI.device, [p.device for p in self.parameters()][0])
        if self.gen_full_image:
            dowOut = [xI]
            for i in xrange(self.g_downsamp_layers+1):
                dowOut.append(self.downsample[3*i+2](self.downsample[3*i+1](self.downsample[3*i](dowOut[-1]))))
            downsamp_out = dowOut[-1]
        else:
            downsamp_out = self.downsample(xI)

        # replicate spatially and concatenate domain information
        if feat is not None:
            feat = feat.unsqueeze(2).unsqueeze(3)
            feat = feat.expand(feat.size(0), feat.size(1), downsamp_out.size(2), downsamp_out.size(3))

            genInp = torch.cat([downsamp_out, feat], dim=1)
        else:
            genInp = downsamp_out
        #net_out = self.generate(genInp)
        outs = [genInp]
        feat_out = []
        d_count = -2
        for i,l in enumerate(self.generate):
            if type(l) is not AdaptiveScaleTconv:
                outs.append(l(outs[i]))
            else:
                deform_out, deform_params = l(outs[i], extra_inp = None if not self.gen_full_image else dowOut[d_count])
                d_count = d_count-1
                outs.append(deform_out)
                feat_out.append(deform_params)
        outImg = outs[-1][:,:,:w,:h]
        if not out_diff:
            return outImg
        else:
            return outImg, feat_out


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, init_stride=2, classify_branch=1, max_filters=None, nc=3, use_bnorm=False, d_kernel_size = 4, patch_size = 2, use_tv_inp = 0):
        super(Discriminator, self).__init__()

        layers = []
        self.use_tv_inp = use_tv_inp
        if self.use_tv_inp:
            self.tvWeight = torch.zeros((1,3,3,3))
            self.tvWeight[0,:,1,1] = -2.0
            self.tvWeight[0,:,1,2] = 1.0; self.tvWeight[0,:,2,1] = 1.0;
            self.tvWeight = Variable(self.tvWeight,requires_grad=False).cuda()
        # Start training
        self.nc=nc + use_tv_inp
        # UGLY HACK
        dkz = d_kernel_size if d_kernel_size > 1 else 4
        if dkz == 3:
            layers.append(nn.Conv2d(self.nc, conv_dim, kernel_size=3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            layers.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=dkz, stride=init_stride, padding=1))
        else:
            layers.append(nn.Conv2d(self.nc, conv_dim, kernel_size=dkz, stride=init_stride, padding=1))
        if use_bnorm:
            layers.append(nn.BatchNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.1, inplace=True))

        curr_dim = conv_dim
        assert(patch_size <= 64)
        n_downSamp = int(np.log2(image_size// patch_size))
        for i in range(1, repeat_num):
            out_dim =  curr_dim*2 if max_filters is None else min(curr_dim*2, max_filters)
            stride = 1 if i >= n_downSamp else 2
            layers.append(nn.Conv2d(curr_dim, out_dim, kernel_size=dkz, stride=stride, padding=1))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            curr_dim = out_dim

        k_size = int(image_size / np.power(2, repeat_num)) + 2- init_stride
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.classify_branch = classify_branch
        if classify_branch==1:
            self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)
        elif classify_branch == 2:
            # This is the projection discriminator!
            #self.embLayer = nn.utils.weight_norm(nn.Linear(c_dim, curr_dim, bias=False))
            self.embLayer = nn.Linear(c_dim, curr_dim, bias=False)

    def forward(self, x, label=None):
        if self.use_tv_inp:
            tvImg = torch.abs(F.conv2d(F.pad(x,(1,1,1,1),mode='replicate'),self.tvWeight))
            x = torch.cat([x,tvImg],dim=1)
        sz = x.size()
        h = self.main(x)
        out_real = self.conv1(h)
        if self.classify_branch==1:
            out_aux = self.conv2(h)
            return out_real.view(sz[0],-1), out_aux.squeeze()
        elif self.classify_branch==2:
            lab_emb = self.embLayer(label)
            out_aux = (lab_emb * (F.normalize(F.avg_pool2d(h,2).view(sz[0], -1), dim=1))).sum(dim=1)
            return (F.avg_pool2d(out_real,2).view(sz[0]) + out_aux).view(-1,1), None
            #return (F.avg_pool2d(out_real,2).squeeze()).view(-1,1)
        else:
            return out_real.squeeze(), None

class Discriminator_SN(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, init_stride=2):
        super(Discriminator_SN, self).__init__()

        layers = []
        layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=init_stride, padding=1)))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(SpectralNorm(nn.Conv2d(curr_dim,min(curr_dim * 2, 1024) , kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = min(curr_dim * 2, 1024)

        k_size = int(image_size / np.power(2, repeat_num)) + 2- init_stride
        self.main = nn.Sequential(*layers)
        self.conv1 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv2 = SpectralNorm(nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False))

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        return out_real.squeeze(), out_aux.squeeze()

class DiscriminatorSmallPatch(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        return out_real.squeeze(), out_aux.squeeze()

class DiscriminatorGAP(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=3, init_stride=1,  max_filters=None, nc=3, use_bnorm=False):
        super(DiscriminatorGAP, self).__init__()

        layers = []
        self.nc=nc
        self.c_dim = c_dim
        layers.append(nn.Conv2d(nc, conv_dim, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1))
        layers.append(nn.MaxPool2d(2))
        if use_bnorm:
            layers.append(nn.BatchNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.1, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            out_dim =  curr_dim*2 if max_filters is None else min(curr_dim*2, max_filters)
            layers.append(nn.Conv2d(curr_dim, out_dim, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            layers.append(ResidualBlockBnorm(dim_in=out_dim, dilation=1, padtype='zero'))
            if (i < 4):
                # We want to have 8x8 resolution vefore GAP input
                layers.append(nn.MaxPool2d(2))
            curr_dim = out_dim

        self.main = nn.Sequential(*layers)
        self.globalPool = nn.AdaptiveAvgPool2d(1)
        self.classifyFC = nn.Linear(curr_dim, c_dim, bias=False)

    def forward(self, x, label=None):
        bsz = x.size(0)
        sz = x.size()
        h = self.main(x)
        out_aux = self.classifyFC(self.globalPool(h).view(bsz, -1))
        return None, out_aux.view(bsz,self.c_dim)

class DiscriminatorGAP_ImageNet(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, c_dim = 5, net_type='vgg19', max_filters=None, global_pool='mean',use_bias=False, class_ftune = 0):
        super(DiscriminatorGAP_ImageNet, self).__init__()

        layers = []
        nFilt = 512 if max_filters is None else max_filters
        self.pnet = Vgg19(only_last=True) if net_type == 'vgg19' else None
        if class_ftune > 0.:
            pAll = list(self.pnet.named_parameters())
            # Multiply by two for weight and bias
            for pn in pAll[::-1][:2*class_ftune]:
                pn[1].requires_grad = True
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(512, nFilt, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(nFilt))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(nFilt, nFilt, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(nFilt))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.layers = nn.Sequential(*layers)
        self.globalPool = nn.AdaptiveAvgPool2d(1) if global_pool == 'mean' else nn.AdaptiveMaxPool2d(1)
        self.classifyFC = nn.Linear(nFilt, c_dim, bias=use_bias)
        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1), requires_grad=False).cuda()
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1), requires_grad=False).cuda()
        self.c_dim  = c_dim

    def forward(self, x, label=None, get_feat = False):
        bsz = x.size(0)
        sz = x.size()
        x = (x - self.shift.expand_as(x))/self.scale.expand_as(x)
        vOut = self.pnet(x)
        h = self.layers(vOut)
        out_aux = self.classifyFC(self.globalPool(h).view(bsz, -1))
        if get_feat:
            return None, out_aux.view(bsz,self.c_dim), h
        else:
            return None, out_aux.view(bsz,self.c_dim)

class DiscriminatorGAP_ImageNet_Weldon(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, c_dim = 5, net_type='vgg19', max_filters=None, global_pool='mean', topk=3, mink=3, use_bias=False):
        super(DiscriminatorGAP_ImageNet_Weldon, self).__init__()

        layers = []
        self.topk = topk
        self.mink = mink
        nFilt = 512 if max_filters is None else max_filters
        self.pnet = Vgg19(only_last=True) if net_type == 'vgg19' else None
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(512, nFilt, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(nFilt))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(nFilt, nFilt, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(nFilt))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.layers = nn.Sequential(*layers)
        #self.AggrConv = nn.conv2d(nFilt, c_dim, kernel_size=1, stride=1, bias=False)
        self.classifyConv = nn.Conv2d(nFilt, c_dim, kernel_size=1, stride=1, bias=use_bias)
        self.globalPool = nn.AdaptiveAvgPool2d(1) if global_pool == 'mean' else nn.AdaptiveMaxPool2d(1)
        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1), requires_grad=False).cuda()
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1), requires_grad=False).cuda()
        self.c_dim  = c_dim

    def forward(self, x, label=None, get_feat = False):
        bsz = x.size(0)
        sz = x.size()
        x = (x - self.shift.expand_as(x))/self.scale.expand_as(x)
        vOut = self.pnet(x)
        h = self.layers(vOut)
        classify_out = self.classifyConv(h)
        if self.topk > 0:
            topk_vals, topk_idx = classify_out.view(bsz,self.c_dim,-1).topk(self.topk)
            out_aux = topk_vals.sum(dim=-1)
            if self.mink > 0:
                mink_vals, mink_idx = classify_out.view(bsz,self.c_dim,-1).topk(self.mink, largest=False)
                out_aux = out_aux + mink_vals.sum(dim=-1)
        else:
            out_aux = self.globalPool(classify_out).view(bsz,-1)
        if get_feat:
            return None, out_aux.view(bsz,self.c_dim), h
        else:
            return None, out_aux.view(bsz,self.c_dim)

class DiscriminatorBBOX(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=64, conv_dim=64, c_dim=5, repeat_num=6):
        super(DiscriminatorBBOX, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_aux = self.conv2(h)
        return out_aux.squeeze()

class DiscriminatorGlobalLocal(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, bbox_size = 64, conv_dim=64, c_dim=5, repeat_num_global=6, repeat_num_local=5, nc=3):
        super(DiscriminatorGlobalLocal, self).__init__()

        maxFilt = 512 if image_size==128 else 128
        globalLayers = []
        globalLayers.append(nn.Conv2d(nc, conv_dim, kernel_size=4, stride=2, padding=1,bias=False))
        globalLayers.append(nn.LeakyReLU(0.2, inplace=True))

        localLayers = []
        localLayers.append(nn.Conv2d(nc, conv_dim, kernel_size=4, stride=2, padding=1, bias=False))
        localLayers.append(nn.LeakyReLU(0.2, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num_global):
            globalLayers.append(nn.Conv2d(curr_dim, min(curr_dim*2,maxFilt), kernel_size=4, stride=2, padding=1, bias=False))
            globalLayers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = min(curr_dim * 2, maxFilt)

        curr_dim = conv_dim
        for i in range(1, repeat_num_local):
            localLayers.append(nn.Conv2d(curr_dim, min(curr_dim * 2, maxFilt), kernel_size=4, stride=2, padding=1, bias=False))
            localLayers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = min(curr_dim * 2, maxFilt)

        k_size_local = int(bbox_size/ np.power(2, repeat_num_local))
        k_size_global = int(image_size/ np.power(2, repeat_num_global))

        self.mainGlobal = nn.Sequential(*globalLayers)
        self.mainLocal = nn.Sequential(*localLayers)

        # FC 1 for doing real/fake
        self.fc1 = nn.Linear(curr_dim*(k_size_local**2+k_size_global**2), 1, bias=False)

        # FC 2 for doing classification only on local patch
        if c_dim > 0:
            self.fc2 = nn.Linear(curr_dim*(k_size_local**2), c_dim, bias=False)
        else:
            self.fc2 = None

    def forward(self, x, boxImg, classify=False):
        bsz = x.size(0)
        h_global = self.mainGlobal(x)
        h_local = self.mainLocal(boxImg)
        h_append = torch.cat([h_global.view(bsz,-1), h_local.view(bsz,-1)], dim=-1)
        out_rf = self.fc1(h_append)
        out_cls = self.fc2(h_local.view(bsz,-1)) if classify and (self.fc2 is not None) else None
        return out_rf.squeeze(), out_cls, h_append

class DiscriminatorGlobalLocal_SN(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, bbox_size = 64, conv_dim=64, c_dim=5, repeat_num_global=6, repeat_num_local=5, nc=3):
        super(DiscriminatorGlobalLocal_SN, self).__init__()

        maxFilt = 512 if image_size==128 else 128
        globalLayers = []
        globalLayers.append(SpectralNorm(nn.Conv2d(nc, conv_dim, kernel_size=4, stride=2, padding=1,bias=False)))
        globalLayers.append(nn.LeakyReLU(0.2, inplace=True))

        localLayers = []
        localLayers.append(SpectralNorm(nn.Conv2d(nc, conv_dim, kernel_size=4, stride=2, padding=1, bias=False)))
        localLayers.append(nn.LeakyReLU(0.2, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num_global):
            globalLayers.append(SpectralNorm(nn.Conv2d(curr_dim, min(curr_dim*2,maxFilt), kernel_size=4, stride=2, padding=1, bias=False)))
            globalLayers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = min(curr_dim * 2, maxFilt)

        curr_dim = conv_dim
        for i in range(1, repeat_num_local):
            localLayers.append(SpectralNorm(nn.Conv2d(curr_dim, min(curr_dim * 2, maxFilt), kernel_size=4, stride=2, padding=1, bias=False)))
            localLayers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = min(curr_dim * 2, maxFilt)

        k_size_local = int(bbox_size/ np.power(2, repeat_num_local))
        k_size_global = int(image_size/ np.power(2, repeat_num_global))

        self.mainGlobal = nn.Sequential(*globalLayers)
        self.mainLocal = nn.Sequential(*localLayers)

        # FC 1 for doing real/fake
        self.fc1 = SpectralNorm(nn.Linear(curr_dim*(k_size_local**2+k_size_global**2), 1, bias=False))

        # FC 2 for doing classification only on local patch
        self.fc2 = SpectralNorm(nn.Linear(curr_dim*(k_size_local**2), c_dim, bias=False))

    def forward(self, x, boxImg, classify=False):
        bsz = x.size(0)
        h_global = self.mainGlobal(x)
        h_local = self.mainLocal(boxImg)
        h_append = torch.cat([h_global.view(bsz,-1), h_local.view(bsz,-1)], dim=-1)
        out_rf = self.fc1(h_append)
        out_cls = self.fc2(h_local.view(bsz,-1)) if classify else None
        return out_rf.squeeze(), out_cls, h_append

class BoxFeatEncoder(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size = 64, k = 4, conv_dim=64, feat_dim=512, repeat_num=5, c_dim=0, norm_type='drop', nc=3):
        super(BoxFeatEncoder, self).__init__()

        maxFilt = 512 if image_size==64 else 128

        layers = []
        layers.append(nn.Conv2d(nc+c_dim, conv_dim, kernel_size=k, stride=2, padding=1))
        if norm_type == 'instance':
            layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        if norm_type == 'drop':
            layers.append(nn.Dropout(p=0.25))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, min(curr_dim*2,maxFilt), kernel_size=k, stride=2, padding=1))
            if norm_type == 'instance':
                layers.append(nn.InstanceNorm2d(min(curr_dim*2,maxFilt), affine=True))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            if norm_type == 'drop':
                layers.append(nn.Dropout(p=0.25))
            curr_dim = min(curr_dim * 2, maxFilt)

        k_size = int(image_size/ np.power(2, repeat_num))

        #layers.append(nn.Dropout(p=0.25))
        self.main= nn.Sequential(*layers)

        # FC 1 for doing real/fake
        self.fc1 = nn.Linear(curr_dim*(k_size**2), feat_dim, bias=False)

    def forward(self, x, c=None):
        bsz = x.size(0)
        if c is not None:
            c = c.unsqueeze(2).unsqueeze(3)
            c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
            xcat = torch.cat([x, c], dim=1)
        else:
            xcat = x

        h= self.main(xcat)
        out_feat = self.fc1(h.view(bsz,-1))
        return out_feat


class BoxFeatGenerator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size = 128, k = 4, conv_dim=64, feat_dim=512, repeat_num=6, c_dim=0, use_residual=0, nc=3):
        super(BoxFeatGenerator, self).__init__()

        maxFilt = 512 if image_size==128 else 128
        layers = []
        fclayers = []
        layers.extend(get_conv_inorm_relu_block(nc+1+c_dim, conv_dim, 7, 1, 3, slope=0.01, padtype='zero'))
        curr_dim = conv_dim
        if use_residual:
            layers.extend(get_conv_inorm_relu_block(conv_dim, conv_dim*2, 3, 1, 1, slope=0.01, padtype='zero'))
            layers.append(nn.MaxPool2d(2))
            # Down-Sampling
            curr_dim = conv_dim*2

        dilation=1
        for i in range(use_residual, repeat_num):
            if use_residual:
                layers.append(ResidualBlock(dim_in=curr_dim, dilation=dilation, padtype='zero'))
                layers.append(nn.MaxPool2d(2))
            else:
                layers.extend(get_conv_inorm_relu_block(curr_dim,  min(curr_dim*2,maxFilt), k, 2, 1, slope=0.01, padtype='zero'))
                curr_dim = min(curr_dim * 2, maxFilt)
            if i > 2:
                dilation = dilation*2

        k_size = int(image_size/ np.power(2, repeat_num))

        #layers.append(nn.Dropout(p=0.25))
        self.main= nn.Sequential(*layers)

        fclayers.append(nn.Linear(curr_dim*(k_size**2), feat_dim*2, bias=False))
        fclayers.append(nn.LeakyReLU(0.01,inplace=True))

        # FC 1 for doing real/fake
        fclayers.append(nn.Linear(feat_dim*2, feat_dim, bias=False))
        self.fc1 = nn.Sequential(*fclayers)

    def forward(self, x, c=None):
        bsz = x.size(0)
        if c is not None:
            c = c.unsqueeze(2).unsqueeze(3)
            c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
            xcat = torch.cat([x, c], dim=1)
        else:
            xcat = x

        h= self.main(xcat)
        out_feat = self.fc1(h.view(bsz,-1))
        return out_feat



##-------------------------------------------------------
## Implementing perceptual loss using VGG19
##-------------------------------------------------------

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

def normalize_tensor(in_feat,eps=1e-10):
    # norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3]).repeat(1,in_feat.size()[1],1,1)
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3])
    return in_feat/(norm_factor.expand_as(in_feat)+eps)

class squeezenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = models.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2,5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])


        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        out = [h_relu1,h_relu2,h_relu3,h_relu4,h_relu5,h_relu6,h_relu7]

        return out


class VGGLoss(nn.Module):
    def __init__(self, network = 'vgg', use_perceptual=True, imagenet_norm = False, use_style_loss=0):
        super(VGGLoss, self).__init__()
        self.criterion = nn.L1Loss()

        self.use_style_loss = use_style_loss

        if network == 'vgg':
            self.chns = [64,128,256,512,512]
        else:
            self.chns = [64,128,256,384,384,512,512]

        if use_perceptual:
            self.use_perceptual = True
            self.lin0 = NetLinLayer(self.chns[0],use_dropout=False)
            self.lin1 = NetLinLayer(self.chns[1],use_dropout=False)
            self.lin2 = NetLinLayer(self.chns[2],use_dropout=False)
            self.lin3 = NetLinLayer(self.chns[3],use_dropout=False)
            self.lin4 = NetLinLayer(self.chns[4],use_dropout=False)
            self.lin0.cuda()
            self.lin1.cuda()
            self.lin2.cuda()
            self.lin3.cuda()
            self.lin4.cuda()

        # Do this since the tensors have already been normalized to have mean and variance [0.5,0.5,0.5]
        self.imagenet_norm = imagenet_norm
        if not self.imagenet_norm:
            self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1)).cuda()
            self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1)).cuda()

        self.net_type = network
        if network == 'vgg':
            self.pnet = Vgg19().cuda()
            self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        else:
            self.pnet = squeezenet().cuda()
            self.weights = [1.0]*7
            if use_perceptual:
                self.lin5 = NetLinLayer(self.chns[5],use_dropout=False)
                self.lin6 = NetLinLayer(self.chns[6],use_dropout=False)
                self.lin5.cuda()
                self.lin6.cuda()
        if self.use_perceptual:
            self.load_state_dict(torch.load('/BS/rshetty-wrk/work/code/controlled-generation/trained_models/perceptualSim/'+network+'.pth'), strict=False)
        for param in self.parameters():
            param.requires_grad = False

    def gram(self, x):
        a, b, c, d = x.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def forward(self, x, y):
        x, y = x.expand(x.size(0), 3, x.size(2), x.size(3)), y.expand(y.size(0), 3, y.size(2), y.size(3))
        if not self.imagenet_norm:
            x = (x - self.shift.expand_as(x))/self.scale.expand_as(x)
            y = (y - self.shift.expand_as(y))/self.scale.expand_as(y)

        x_vgg, y_vgg = self.pnet(x), self.pnet(y)
        loss = 0
        if self.use_perceptual:
            normed_x = [normalize_tensor(x_vgg[kk]) for (kk, out0) in enumerate(x_vgg)]
            normed_y = [normalize_tensor(y_vgg[kk]) for (kk, out0) in enumerate(y_vgg)]
            diffs = [(normed_x[kk]-normed_y[kk].detach())**2 for (kk,out0) in enumerate(x_vgg)]
            loss = self.lin0.model(diffs[0]).mean()
            loss = loss + self.lin1.model(diffs[1]).mean()
            loss = loss + self.lin2.model(diffs[2]).mean()
            loss = loss + self.lin3.model(diffs[3]).mean()
            loss = loss + self.lin4.model(diffs[4]).mean()
            if(self.net_type=='squeeze'):
                loss = loss + self.lin5.model(diffs[5]).mean()
                loss = loss + self.lin6.model(diffs[6]).mean()
            if self.use_style_loss:
                style_loss = 0.
                for kk in xrange(3, len(x_vgg)):
                    style_loss += self.criterion(self.gram(x_vgg[kk]), self.gram(y_vgg[kk]))
                loss += self.use_style_loss * style_loss
        else:
            for i in range(len(x_vgg)):
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

