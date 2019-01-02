import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from removalmodels.models import Generator
from removalmodels.models import GeneratorDiff, GeneratorDiffWithInp, GeneratorDiffAndMask, GeneratorDiffAndMask_V2, GeneratorBoxReconst, GeneratorOnlyMask, GeneratorMaskAndFeat, GeneratorMaskAndFeat_ImNetBackbone, GeneratorMaskAndFeat_ImNetBackbone_V2
from removalmodels.models import Discriminator_SN, Discriminator, DiscriminatorBBOX, DiscriminatorGlobalLocal, DiscriminatorGlobalLocal_SN, DiscriminatorGAP, DiscriminatorGAP_ImageNet
from removalmodels.models import BoxFeatEncoder, BoxFeatGenerator, VGGLoss
from PIL import Image
import gc
import random
from utils.utils import show
from collections import OrderedDict


###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def save_checkpoint(state, fappend ='dummy', outdir = 'cv'):
    filename = os.path.join(outdir,'checkpoint_stargan_'+fappend)
    torch.save(state, filename)

class Solver(object):

    def __init__(self, celebA_loader, rafd_loader, config, mode='train', pretrainedcv=None, mask_loader = None):
        # Data loader
        self.celebA_loader = celebA_loader
        self.rafd_loader = rafd_loader
        self.arch = vars(config)
        self.differential_generator = config.differential_generator
        self.mask_loader = mask_loader
        self.use_maskprior_gan = getattr(config, 'use_maskprior_gan', 0)
        self.use_gtmask_inp = getattr(config, 'use_gtmask_inp', 0)
        self.boxprior = getattr(config, 'boxprior', 0)
        self.maskprior_matchclass = getattr(config, 'maskprior_matchclass', 0)
        self.alternate_mask_train= getattr(config, 'alternate_mask_train', 0)
        self.train_g_every = getattr(config, 'train_g_every', 2)

        self.mode = mode
        # Model hyper-parameters
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.box_size = config.box_size
        self.boxfeat_dim = config.boxfeat_dim
        self.g_conv_dim = config.g_conv_dim
        self.g_smooth_layers = config.g_smooth_layers
        self.g_downsamp_layers= getattr(config, 'g_downsamp_layers', 2)
        self.g_dil_start = config.g_dil_start
        self.g_repeat_num = config.g_repeat_num
        self.g_binary_mask= config.g_binary_mask
        self.g_upsample_type= getattr(config, 'g_upsample_type', 't_conv')
        self.g_nupsampFilters= getattr(config, 'g_nupsampFilters', 1)
        self.g_pad_type = getattr(config, 'g_pad_type', 'zeros')
        self.g_fixed_size= getattr(config, 'g_fixed_size', 0)
        self.gen_fullimage = getattr(config, 'gen_fullimage', 0)

        self.m_upsample_type = getattr(config, 'm_upsample_type', 'bilinear')

        self.d_use_spectralnorm= getattr(config, 'd_use_spectralnorm', 0)
        self.d_max_filters = getattr(config, 'd_max_filters', None)
        self.d_max_filters_cls = getattr(config, 'd_max_filters_cls', None)
        self.d_global_pool = getattr(config, 'd_global_pool', 'mean')
        self.d_kernel_size = getattr(config, 'd_kernel_size', 4)
        self.d_conv_dim = config.d_conv_dim
        self.d_repeat_num = config.d_repeat_num
        self.e_repeat_num = getattr(config, 'e_repeat_num',self.d_repeat_num-1)
        self.d_init_stride= config.d_init_stride
        self.d_train_repeat = config.d_train_repeat
        self.d_local_supervision = getattr(config, 'd_local_supervision', 0)
        self.d_patch_size = getattr(config, 'd_patch_size', 2)

        self.e_norm_type = config.e_norm_type
        self.e_ksize = config.e_ksize
        self.lowres_mask = getattr(config, 'lowres_mask',0)
        self.mask_size = getattr(config, 'mask_size',32)
        self.mask_additional_cond = getattr(config, 'mask_additional_cond','image')
        self.perclass_mask = getattr(config, 'perclass_mask',0)
        self.mask_noinplabel = getattr(config, 'mask_noinplabel',0)
        self.mask_normalize_byclass = getattr(config, 'mask_normalize_byclass',0)

        self.use_seperate_classifier = config.use_seperate_classifier
        self.use_gap_classifier = getattr(config, 'use_gap_classifier',0)
        self.use_imagenet_pretrained = getattr(config, 'use_imagenet_pretrained',None)
        self.use_imagenet_pretrained_mask = getattr(config, 'use_imagenet_pretrained_mask',None)
        self.cond_inp_pnet = getattr(config, 'cond_inp_pnet',0)
        self.cond_parallel_track = getattr(config, 'cond_parallel_track',0)
        self.use_imnetmask_v2= getattr(config, 'use_imnetmask_v2',0)
        self.use_bnorm= getattr(config, 'use_bnorm',0)
        self.use_bnorm_mask = getattr(config, 'use_bnorm_mask',0) # 0 is no norm, 1 is batch norm, 2 is instance norm
        self.adv_classifier = config.adv_classifier
        self.adv_loss_type = getattr(config, 'adv_loss_type', 'wgan')
        self.m_adv_loss_type = getattr(config, 'm_adv_loss_type', 'wgan')

        self.train_boxreconst = config.train_boxreconst
        self.full_image_encoder = getattr(config,'full_image_encoder', 0)
        self.only_reconst_loss = getattr(config,'only_reconst_loss',0)
        self.use_box_label = getattr(config,'use_box_label',0)
        self.g_fine_tune= getattr(config,'g_fine_tune',0)
        self.no_inpainter = getattr(config,'no_inpainter',0)
        self.fixed_m = getattr(config,'fixed_m',0)
        self.only_gt_mask = getattr(config,'only_gt_mask',0)
        self.train_only_g= getattr(config,'train_only_g',0)
        if self.train_only_g:
            self.only_gt_mask = 1
        self.dilateMask = getattr(config,'dilateMask',0)
        self.train_g_wo_m = getattr(config, 'train_g_wo_m',0)
        self.fixed_classifier = getattr(config,'fixed_classifier',0)
        self.train_only_d = getattr(config,'train_only_d',0)
        self.train_robust_d= getattr(config,'train_robust_d',0)
        self.compositional_loss= getattr(config,'compositional_loss',0)
        self.use_random_boxes= getattr(config,'use_random_boxes',0)
        self.use_past_masks = getattr(config,'use_past_masks',0)
        self.use_tv_inp = getattr(config,'use_tv_inp',0)
        self.only_random_boxes_discr = getattr(config,'only_random_boxes_discr',0)
        self.discrim_masked_image = getattr(config,'discrim_masked_image',0)


        # Hyper-parameteres
        self.use_topk_patch = config.use_topk_patch
        self.lambda_cls = config.lambda_cls
        self.onlytargetclsloss = getattr(config,'onlytargetclsloss',0)
        self.useclsweights = getattr(config,'useclsweights',0)
        self.clsweight_scale_decay= getattr(config,'clsweight_scale_decay',0)
        self.clsweight_scale_init = getattr(config,'clsweight_scale_init',0)
        self.lambda_rec = config.lambda_rec
        self.lambda_feat_match = getattr(config, 'lambda_feat_match', 0)
        self.lambda_vggloss = getattr(config, 'lambda_vggloss', 0)
        self.use_style_loss = getattr(config, 'use_style_loss', 0)
        self.lambda_smoothloss= getattr(config, 'lambda_smoothloss', 0)
        self.lambda_tvloss= getattr(config, 'lambda_tvloss', 0.)
        self.lambda_maskL1loss= getattr(config, 'lambda_maskL1loss', 0)
        self.grad_weighted_l1loss= getattr(config, 'grad_weighted_l1loss', 0)
        self.lambda_maskfake_loss= getattr(config, 'lambda_maskfake_loss', 1.)
        self.vggloss_nw = getattr(config, 'vggloss_nw', 'vgg')
        self.onlymaskgen= getattr(config, 'onlymaskgen', 0)

        self.e_addnoise = getattr(config, 'e_addnoise', 0)
        self.e_masked_image= getattr(config, 'e_masked_image', 0)
        self.e_use_residual= getattr(config, 'e_use_residual', 0)
        self.e_bias = getattr(config, 'e_bias', False)
        self.load_encoder= getattr(config, 'load_encoder', 1)
        self.load_generator= getattr(config, 'load_generator', 1)
        self.load_discriminator = getattr(config, 'load_discriminator', 1)


        self.adv_rec = config.adv_rec
        self.lambda_gp = config.lambda_gp
        self.g_lr = config.g_lr
        self.e_lr = 1.0* getattr(config, 'e_lr', self.g_lr)
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.dataset = config.dataset
        self.nc = 1 if self.dataset == 'mnist' else 3
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.decay_every = getattr(config, 'decay_every', 5)
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model

        # Test settings
        self.test_model = config.test_model

        # Path
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.model_save_path = config.model_save_path
        self.result_path = config.result_path

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.selected_attrs = config.selected_attrs

        # Build tensorboard if use
        self.build_model(pretrainedcv)
        if self.use_tensorboard:
            self.build_tensorboard()

        if config.onlypretrained_discr and (self.mode=='train' or self.mode=='eval'):
            self.load_pretrained_discr(config.onlypretrained_discr)
        if getattr(config, 'onlypretrained_encoder',None):
            self.load_pretrained_encoder(config.onlypretrained_encoder)
        if getattr(config, 'onlypretrained_generator',None):
            cvG = torch.load(config.onlypretrained_generator)
            self.load_pretrained_generator(cvG)

    def build_model(self, pretrainedcv):
        # Define a generator and a discriminator
        #-------------------------------
        # Initialize the image generator
        #-------------------------------
        if self.train_boxreconst:
            #------------------------------------------------------------------------------------------------
            # This is the box generation mode
            #------------------------------------------------------------------------------------------------
            GenClass = GeneratorBoxReconst
            self.G = GenClass(self.g_conv_dim, self.boxfeat_dim, self.g_repeat_num, self.g_downsamp_layers,
                              self.g_dil_start, self.g_upsample_type, self.g_pad_type, nc=self.nc,
                              n_upsamp_filt=self.g_nupsampFilters, gen_full_image=self.gen_fullimage)
        else:
            #--------------------------------------------------------------------------
            # This is the full image (could be differential and masked) generation mode
            #--------------------------------------------------------------------------
            GenClass = GeneratorOnlyMask if self.onlymaskgen==1 else Generator if not self.differential_generator else GeneratorDiff if self.differential_generator==1 else GeneratorDiffWithInp if self.differential_generator==2 else GeneratorDiffAndMask if self.differential_generator==3 else GeneratorDiffAndMask_V2
            self.G = GenClass(self.g_conv_dim, self.c_dim, self.g_repeat_num, self.g_smooth_layers, self.g_binary_mask)

        #-------------------------------
        # Initialize the real/fake discriminator and classifier
        #-------------------------------
        if (self.mode == 'train' or self.mode=='eval'):
            if (self.train_boxreconst==1 or self.train_boxreconst==2):
                DisClass = DiscriminatorGlobalLocal_SN if self.d_use_spectralnorm else DiscriminatorGlobalLocal
                self.D = DisClass(self.image_size, self.box_size, self.d_conv_dim, self.c_dim, self.d_repeat_num,
                                  self.d_repeat_num-1, nc=self.nc, d_kernel_size=self.d_kernel_size)
            else:
                DisClass = Discriminator_SN if self.d_use_spectralnorm else Discriminator
                self.D = DisClass(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num,
                                  self.d_init_stride, max_filters = self.d_max_filters, nc = self.nc*(1+self.discrim_masked_image),
                                  d_kernel_size=self.d_kernel_size, patch_size = self.d_patch_size, use_tv_inp=self.use_tv_inp)
                if self.use_seperate_classifier and not self.train_only_g:
                    if self.use_imagenet_pretrained is not None:
                        self.D_cls = DiscriminatorGAP_ImageNet(self.image_size, self.c_dim, net_type = self.use_imagenet_pretrained, max_filters = self.d_max_filters_cls, global_pool = self.d_global_pool)
                    else:
                        DisClass = DiscriminatorGAP if self.use_gap_classifier else DisClass
                        self.D_cls = DisClass(self.image_size, self.d_conv_dim, self.c_dim, 4 if self.use_gap_classifier else self.d_repeat_num ,
                                          self.d_init_stride, max_filters = self.d_max_filters_cls, use_bnorm=self.use_bnorm)
                else:
                    self.D_cls = None
        else:
            self.D = None
            self.D_cls = None

        if not self.train_only_g:
            if ((self.train_boxreconst == 1) and not self.full_image_encoder) and (self.boxfeat_dim > 0):
                #-----------------------------------------------------------------------------
                # Initialize the box image encoder to pre-train the generator as an autoencoder
                #-----------------------------------------------------------------------------
                self.E = BoxFeatEncoder(self.box_size, self.e_ksize, self.d_conv_dim, self.boxfeat_dim,
                                        self.e_repeat_num, (self.use_box_label*self.c_dim),
                                        self.e_norm_type, nc=self.nc)
            elif ((self.train_boxreconst == 2) or self.full_image_encoder) and (self.boxfeat_dim > 0):
                #-----------------------------------------------------------------------------
                # Initialize the feature generator network to do editing (adding/removing)
                #-----------------------------------------------------------------------------
                self.E = BoxFeatGenerator(self.image_size, self.e_ksize, self.d_conv_dim, self.boxfeat_dim,
                                          c_dim=(self.use_box_label*self.c_dim)+(self.e_masked_image>1)*3,
                                          use_residual=self.e_use_residual, nc=self.nc)
            elif (self.train_boxreconst == 3):
                #-----------------------------------------------------------------------------
                # Initialize the mask and feature generator network to do editing (adding/removing)
                #-----------------------------------------------------------------------------
                #def __init__(self, conv_dim=64, c_dim=5, repeat_num=5, g_smooth_layers=0, binary_mask=0, out_feat_dim=256, up_sampling_type='bilinear', n_upsamp_filt=2):
                maskgen =  GeneratorMaskAndFeat if self.use_imagenet_pretrained_mask is None else GeneratorMaskAndFeat_ImNetBackbone_V2 if self.use_imnetmask_v2 else GeneratorMaskAndFeat_ImNetBackbone
                self.E = maskgen(self.g_conv_dim, self.c_dim, self.e_repeat_num, 0,binary_mask = self.g_binary_mask,
                                              out_feat_dim = self.boxfeat_dim, mask_size = self.mask_size if self.image_size == 128 else self.mask_size//2,
                                              additional_cond = self.mask_additional_cond, per_classMask = self.perclass_mask,
                                              noInpLabel= self.mask_noinplabel, mask_normalize = self.mask_normalize_byclass, nc = 4 if self.use_gtmask_inp else 3,
                                              use_bias = self.e_bias, use_bnorm = self.use_bnorm_mask, cond_inp_pnet=self.cond_inp_pnet,
                                              cond_parallel_track = self.cond_parallel_track)
            else:
                self.E = None
        else:
            self.E = None

        if self.lambda_vggloss:
            self.vggLoss = VGGLoss(network=self.vggloss_nw, imagenet_norm = False, use_style_loss = self.use_style_loss)

        if self.use_maskprior_gan:
            classify_type = 0 if self.c_dim ==1 else 2
            self.mask_D = Discriminator(self.mask_size, self.d_conv_dim, self.c_dim,  int(np.log2(self.mask_size) - 1),
                                        self.d_init_stride, classify_branch = classify_type, max_filters = 256 if self.c_dim == 1 else 512,
                                        nc=1)


        if self.g_upsample_type!= 'deform':
            self.G.apply(weights_init)
        if self.mode == 'train' and not self.d_use_spectralnorm:
            self.D.apply(weights_init)
        if (self.D_cls is not None) and self.mode == 'train' and (not self.d_use_spectralnorm) and self.use_seperate_classifier and (self.use_imagenet_pretrained is None):
            self.D_cls.apply(weights_init)
        if (self.E is not None) and (self.use_imagenet_pretrained_mask is None):
            self.E.apply(weights_init)
            #if self.mask_normalize_byclass:
            #    self.E.final_Layer_mask.bias.data[-1] = 3.
        if self.use_maskprior_gan:
            self.mask_D.apply(weights_init)

        # Optimizers
        if self.mode == 'train':
            if self.g_upsample_type== 'deform':
                #split the parameters into groups
                g_params = [{'params':[p[1]  for p in self.G.named_parameters() if 'coordfilter' not in p[0]]}]
                g_params.append({'params':[p[1]  for p in self.G.named_parameters() if 'coordfilter' in p[0]], 'lr':self.g_lr*0.1})
            else:
                g_params = self.G.parameters()


            self.g_optimizer = torch.optim.Adam(g_params, self.g_lr, [self.beta1, self.beta2])#, amsgrad=True)
            self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])#, amsgrad=True)
            if (self.D_cls is not None) and self.use_seperate_classifier:
                self.dcls_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_cls.parameters()), self.d_lr, [self.beta1, self.beta2], weight_decay=0.01)#, amsgrad=True)
            if self.E is not None:
                self.e_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.E.parameters()), self.e_lr, [self.beta1, self.beta2])#, amsgrad=True)
            if self.use_maskprior_gan:
                self.mask_d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.mask_D.parameters()), self.d_lr, [self.beta1, self.beta2])#, amsgrad=True)

            # Print networks
            self.print_network(self.G, 'G')
            self.print_network(self.D, 'D')
            if (self.D_cls is not None) and self.use_seperate_classifier:
                self.print_network(self.D_cls, 'D_cls')
            if self.E is not None:
                self.print_network(self.E, 'E')
            if self.use_maskprior_gan:
                self.print_network(self.mask_D, 'mask_D')

            # Before the training begins, save the requires_grad setting for all the params
            for p in self.G.parameters():
                p.requires_grad_orig = bool(p.requires_grad and ((self.train_boxreconst<2) or self.g_fine_tune))#
            for p in self.D.parameters():
                p.requires_grad_orig = p.requires_grad#
            if (self.D_cls is not None) and self.use_seperate_classifier:
                for p in self.D_cls.parameters():
                    p.requires_grad_orig = p.requires_grad#
            if self.E is not None:
                for p in self.E.parameters():
                    p.requires_grad_orig = p.requires_grad#
            if self.use_maskprior_gan:
                for p in self.mask_D.parameters():
                    p.requires_grad_orig = p.requires_grad#

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model(pretrainedcv)

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print 'Using multiple GPUs'
                self.G = nn.DataParallel(self.G)
                #self.vggLoss = nn.DataParallel(self.vggLoss)
            self.G.cuda()
            if (self.mode == 'train' or self.mode=='eval'):
                self.D.cuda()
            if (self.D_cls is not None) and self.use_seperate_classifier and (self.mode == 'train' or self.mode=='eval'):
                self.D_cls.cuda()
            if self.E is not None:
                self.E.cuda()
            if self.use_maskprior_gan and (self.mode == 'train' or self.mode=='eval'):
                self.mask_D.cuda()



    def getImageSizeMask(self, mask):
        return F.upsample(mask, scale_factor=int(self.image_size/mask.size(-1)), mode=self.m_upsample_type) if mask.size(-1)!= self.image_size else mask

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            if p.requires_grad == True:
                num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self, cv=None):
        if cv is None:
            cv = torch.load(self.pretrained_model)
        if 'generator_state_dict' in cv and self.load_generator:
            self.load_pretrained_generator(cv)
        else:
            print 'Generator not found'
        if (self.mode == 'train' or self.mode=='eval') and self.load_discriminator: #and not self.d_use_spectralnorm:
            if 'discriminator_state_dict' in cv:
                self.D.load_state_dict(cv['discriminator_state_dict'])
            else:
                print 'Discriminator not found'
        if (self.E is not None) and self.load_encoder:
            if 'encoder_state_dict' in cv:
                self.E.load_state_dict(cv['encoder_state_dict'])
            else:
                print 'Encoder not found'
        if self.use_maskprior_gan and (self.mode == 'train'):
            if 'mask_discriminator_state_dict' in cv:
                self.mask_D.load_state_dict(cv['mask_discriminator_state_dict'])
            else:
                print 'Mask discriminator not found'
        if (self.D_cls is not None) and self.use_seperate_classifier and self.load_discriminator:
            if 'discriminator_cls_state_dict' in cv:
                self.D_cls.load_state_dict(cv['discriminator_cls_state_dict'])
            else:
                print 'Object classifier not found'
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def load_pretrained_encoder(self, fname):
        cv = torch.load(fname)
        if (self.E is not None):
            self.E.load_state_dict(cv['encoder_state_dict'])

    def load_pretrained_generator(self, cv):
        new_state_dict = OrderedDict()
        for k, v in cv['generator_state_dict'].items():
            # remove `module.`
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        cv['generator_state_dict'] = new_state_dict
        if (self.G is not None):
            self.G.load_state_dict(cv['generator_state_dict'])

    def load_pretrained_discr(self, fname):
        cv = torch.load(fname)
        if 'discriminator_state_dict' in cv:
            self.D.load_state_dict(cv['discriminator_state_dict'])
        if (self.D_cls is not None) and self.use_seperate_classifier and 'discriminator_cls_state_dict' in cv:
            self.D_cls.load_state_dict(cv['discriminator_cls_state_dict'])
        if self.use_maskprior_gan and 'mask_discriminator_state_dict' in cv and (self.mode == 'train'):
            self.mask_D.load_state_dict(cv['mask_discriminator_state_dict'])
        print('loaded pre-trained discr (step: {})..!'.format(fname))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, g_lr, d_lr, e_lr=None):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        if self.use_seperate_classifier:
            for param_group in self.dcls_optimizer.param_groups:
                param_group['lr'] = d_lr
        if (self.E is not None) and (e_lr is not None):
            for param_group in self.e_optimizer.param_groups:
                param_group['lr'] = e_lr
        if self.use_maskprior_gan:
            for param_group in self.mask_d_optimizer.param_groups:
                param_group['lr'] = d_lr
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.e_lr = e_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        if self.use_seperate_classifier:
            self.dcls_optimizer.zero_grad()
        if self.E is not None:
            self.e_optimizer.zero_grad()
        if self.use_maskprior_gan:
            self.mask_d_optimizer.zero_grad()

    def to_var(self, x, volatile=False):
        return Variable(x.cuda() if torch.cuda.is_available() else x , volatile=volatile)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def threshold(self, x):
        return (x>0.5).float()

    def compute_accuracy(self, x, y, dataset):
        x = F.sigmoid(x)
        predicted = self.threshold(x)
        correct = (predicted == y).float()
        accuracy = torch.mean(correct, dim=0) * 100.0
        return accuracy

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def make_coco_labels(self, real_c):
        """Generate domain labels for CelebA for debugging/testing.

        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """
        y = np.eye(real_c.size(1))

        fixed_c_list = []

        # single object addition and removal
        for i in range(2*self.c_dim):
            fixed_c = real_c.clone()
            for c in fixed_c:
                if i%2:
                    c[i//2] = 0.
                else:
                    c[i//2] = 1.
            fixed_c_list.append(self.to_var(fixed_c, volatile=True))

        # multi-attribute transfer (H+G, H+A, G+A, H+G+A)
        #if self.dataset == 'CelebA':
        #    for i in range(4):
        #        fixed_c = real_c.clone()
        #        for c in fixed_c:
        #            if i in [0, 1, 3]:   # Hair color to brown
        #                c[:3] = y[2]
        #            if i in [0, 2, 3]:   # Gender
        #                c[3] = 0 if c[3] == 1 else 1
        #            if i in [1, 2, 3]:   # Aged
        #                c[4] = 0 if c[4] == 1 else 1
        #        fixed_c_list.append(self.to_var(fixed_c, volatile=True))
        return fixed_c_list

    def make_celeb_labels(self, real_c):
        """Generate domain labels for CelebA for debugging/testing.

        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """
        y = [torch.FloatTensor([1, 0, 0]),  # black hair
             torch.FloatTensor([0, 1, 0]),  # blond hair
             torch.FloatTensor([0, 0, 1])]  # brown hair

        fixed_c_list = []

        # single attribute transfer
        for i in range(self.c_dim):
            fixed_c = real_c.clone()
            for c in fixed_c:
                if i < 3:
                    c[:3] = y[i]
                else:
                    c[i] = 0 if c[i] == 1 else 1   # opposite value
            fixed_c_list.append(self.to_var(fixed_c, volatile=True))

        # multi-attribute transfer (H+G, H+A, G+A, H+G+A)
        if self.dataset == 'CelebA':
            for i in range(4):
                fixed_c = real_c.clone()
                for c in fixed_c:
                    if i in [0, 1, 3]:   # Hair color to brown
                        c[:3] = y[2]
                    if i in [0, 2, 3]:   # Gender
                        c[3] = 0 if c[3] == 1 else 1
                    if i in [1, 2, 3]:   # Aged
                        c[4] = 0 if c[4] == 1 else 1
                fixed_c_list.append(self.to_var(fixed_c, volatile=True))
        return fixed_c_list


    def compute_gradient_penalty(self, real_x, fake_x, discrim_type= 'regular', label=None, bbox=None):
        rsz = real_x.shape[0:1] + (1,)*(real_x.ndimension()-1)
        alpha = torch.rand(*rsz).cuda().expand_as(real_x)
        interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)

        if discrim_type == 'regular':
            if bbox is None:
                out, out_cls = self.D(interpolated)
            else:
                out, out_cls, _ = self.D(interpolated, self.getRandBoxWith(interpolated, bbox))
        elif discrim_type == 'seperate':
            out, out_cls = self.D_cls(interpolated)
        elif discrim_type == 'mask':
            out, _ = self.mask_D(interpolated , label = label)

        ## RS HACK
        #interpout = torch.cat([out.view(out.size(0), -1), out_cls],dim=-1)
        if not self.onlymaskgen:
            interpout = out
        else:
            interpout = out_cls

        grad = torch.autograd.grad(outputs=interpout,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(interpout.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        #import ipdb; ipdb.set_trace()
        d_loss_gp = torch.mean((grad_l2norm - 1)**2)

        return d_loss_gp

    def compute_real_fake_loss_local(self, scores, loss_type, label, loss_for='discr', only_pos = False):
        if loss_for == 'discr':
            scores = scores.view(scores.size(0),-1)
            label = (label.data.view(label.size(0),-1)>0.01).float()
            score_p = scores[(1.-label).byte()]
            if only_pos:
                score_p = scores
                score_n = []
                n_n = 0
            else:
                score_n = scores[label.byte()]
                n_n = score_n.numel()
            n_p = score_p.numel()

            d_loss_p = 0.
            d_loss_n = 0.
            if loss_type == 'lsgan':
                # The Loss for least-square gan
                if len(score_p):
                    d_loss_p = torch.pow(score_p - 1., 2).sum()
                if len(score_n):
                    d_loss_n = torch.pow(score_n + 1 , 2).sum()
            elif loss_type == 'wgan':
                # The Loss for Wgan
                if len(score_p):
                    d_loss_p += -(torch.sum(score_p))
                if len(score_n):
                    sign = 1. if not only_pos else -1.
                    d_loss_n += sign*torch.sum(score_n)
            else:
                if len(score_p):
                    d_loss += (F.binary_cross_entropy_with_logits(score_p,torch.ones_like(score_p).detach()))
                if len(score_n):
                    targ = torch.zeros_like(score_n) if not only_pos else torch.ones_like(score_n)
                    d_loss += (F.binary_cross_entropy_with_logits(score_n,targ.detach()))

            return d_loss_p, d_loss_n, n_p, n_n


    def compute_real_fake_loss(self, scores, loss_type, datasrc = 'real', loss_for='discr'):
        if loss_for == 'discr':
            if datasrc == 'real':
                if loss_type == 'lsgan':
                    # The Loss for least-square gan
                    d_loss = torch.pow(scores - 1., 2).mean()
                elif loss_type == 'hinge':
                    # Hinge loss used in the spectral GAN paper
                    d_loss = - torch.mean(torch.clamp(scores-1.,max=0.))
                elif loss_type == 'wgan':
                    # The Loss for Wgan
                    d_loss = - torch.mean(scores)
                else:
                    scores = scores.view(scores.size(0),-1).mean(dim=1)
                    d_loss = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores).detach())
            else:
                if loss_type == 'lsgan':
                    # The Loss for least-square gan
                    d_loss = torch.pow((scores),2).mean()
                elif loss_type == 'hinge':
                    # Hinge loss used in the spectral GAN paper
                    d_loss = -torch.mean(torch.clamp(-scores-1.,max=0.))
                elif loss_type == 'wgan':
                    # The Loss for Wgan
                    d_loss = torch.mean(scores)
                else:
                    scores = scores.view(scores.size(0),-1).mean(dim=1)
                    d_loss = F.binary_cross_entropy_with_logits(scores, torch.zeros_like(scores).detach())

            return d_loss
        else:
            if loss_type == 'lsgan':
                # The Loss for least-square gan
                g_loss = torch.pow(scores - 1., 2).mean()
            elif (loss_type == 'wgan') or (loss_type == 'hinge') :
                g_loss = - torch.mean(scores)
            else:
                scores = scores.view(scores.size(0),-1).mean(dim=1)
                g_loss = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores).detach())
            return g_loss


    def forward_generator(self, real_x, boxImg=None, imagelabel = None, mask=None, boxlabel=None, bbox=None, get_feat=None, onlyMasks = False, mask_threshold = 0.3, withGTMask=False, dilate=None, n_iter=0):

        if self.train_boxreconst == 3:
            fake_x,_,mask = self.forward_fulleditor(real_x, imagelabel, onlyMasks=onlyMasks, mask_threshold = mask_threshold, gtMask=mask, withGTMask=withGTMask, dilate=dilate, n_iter = n_iter)
        elif self.train_boxreconst > 0:
            if get_feat:
                fake_x, feat = self.forward_boxreconst(real_x, boxImg, mask, boxlabel, bbox,get_feat=get_feat)
            else:
                fake_x = self.forward_boxreconst(real_x, boxImg, mask, boxlabel, bbox)
        elif self.train_boxreconst == 0:
                fake_x, mask = self.G(real_x, boxlabel, out_diff = 1)
        return fake_x, mask

    def classify(self, x ):
        if self.use_seperate_classifier:
            return self.D_cls(x)
        else:
            return self.D(x)


    def train(self):

        # Set dataloader
        if self.dataset in ['CelebA', 'coco', 'mnist', 'celebbox', 'pascal']:
            self.data_loader = self.celebA_loader
        else:
            self.data_loader = self.rafd_loader

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        fixed_x = []
        real_c = []
        for i, (images, labels) in enumerate(self.data_loader):
            fixed_x.append(images)
            real_c.append(labels)
            if i == 3:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)
        real_c = torch.cat(real_c, dim=0)

        if self.dataset == 'CelebA':
            fixed_c_list = self.make_celeb_labels(real_c)
        elif self.dataset == 'coco':
            fixed_c_list = self.make_coco_labels(real_c)
        elif self.dataset == 'RaFD':
            fixed_c_list = []
            for i in range(self.c_dim):
                fixed_c = self.one_hot(torch.ones(fixed_x.size(0)) * i, self.c_dim)
                fixed_c_list.append(self.to_var(fixed_c, volatile=True))

        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr
        e_lr = self.e_lr

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[-2])
        else:
            start = 0


        if self.lambda_smoothloss:
            smoothWeight = -1.0/9.*torch.ones((1,1,3,3))
            smoothWeight[0,0,1,1] = 1.0
            smoothWeight = Variable(smoothWeight,requires_grad=False).cuda()

        # Start training
        loss = {}
        accLog = {}
        avgexp= 0.95
        start_time = time.time()
        cocoAndCelebset= set(['CelebA', 'coco', 'pascal'])

        for e in range(start, self.num_epochs):
            for i, (real_x, real_label) in enumerate(self.data_loader):
                # Generat fake labels randomly (target domain labels)
                #plt.imshow(((real_x.numpy()[6,[0,1,2],:,:].transpose(1,2,0)+1.0)*255./2.0).astype(np.uint8)); plt.show()
                cbsz = real_label.size(0)
                if self.dataset == 'coco':
                    rand_idx = np.random.randint(0,real_label.size(1),cbsz)
                    aidx= np.arange(cbsz)
                    fake_label = real_label.clone()
                    # XXX: FIX THIS!!!!!
                    fake_label[aidx,rand_idx] =  1. - real_label[aidx,rand_idx] if not self.arch['only_remove_train'] else 0.
                else:
                    rand_idx = torch.randperm(real_label.size(0))
                    fake_label = real_label[rand_idx]

                for p in self.G.parameters():
                    p.requires_grad = False#

                real_c = real_label.clone()
                fake_c = fake_label.clone() #if (np.random.rand()> 0.5) or (self.dataset=='CelebA') else real_label.clone()

                # Convert tensor to variable
                real_x = self.to_var(real_x)
                real_c = self.to_var(real_c)           # input for the generator
                fake_c = self.to_var(fake_c)
                real_label = self.to_var(real_label)   # this is same as real_c if dataset == 'CelebA'
                fake_label = self.to_var(fake_label)

                # ================== Train D ================== #

                # Compute loss with real images
                out_src, out_cls = self.D(real_x)
                if self.use_seperate_classifier:
                    out_cls = self.D_cls(real_x)

                d_loss_real = self.compute_real_fake_loss(out_src, self.adv_loss_type, datasrc = 'real')

                if self.dataset in cocoAndCelebset:
                    d_loss_cls = F.binary_cross_entropy_with_logits(
                        out_cls.view(cbsz, -1), real_label, size_average=False) / cbsz
                else:
                    d_loss_cls = F.cross_entropy(out_cls, real_label)

                # Compute classification accuracy of the discriminator
                if (i+1) % self.log_step == 0:
                    accuracies = self.compute_accuracy(out_cls.view(cbsz,-1), real_label, self.dataset)
                    accLog['acc'] = accLog['acc'] * 0.8 + (1-0.8)* accuracies.data.cpu().numpy() if 'acc' in accLog else  accuracies.data.cpu().numpy()
                    log = ["{}: {:.2f}".format(self.selected_attrs[cati],acc) for cati,acc in enumerate(accLog['acc'])]
                    print(log)

                # Compute loss with fake images
                fake_x = self.G(real_x, fake_c)
                fake_x = Variable(fake_x.data)
                out_src, out_cls_fake = self.D(fake_x)
                if self.use_seperate_classifier and self.adv_classifier:
                    out_cls_fake = self.D_cls(fake_x)

                d_loss_fake = self.compute_real_fake_loss(out_src, self.adv_loss_type, datasrc = 'fake')

                if self.adv_classifier:
                    d_loss_cls = d_loss_cls + (F.binary_cross_entropy_with_logits(
                        out_cls_fake.view(cbsz,-1), real_label, size_average=False) / cbsz)

                # Backward + Optimize
                if not self.use_seperate_classifier:
                    d_loss = (1.-self.onlymaskgen) * (d_loss_real + d_loss_fake) + self.lambda_cls * d_loss_cls
                else:
                    d_loss = d_loss_real + d_loss_fake

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
                if self.use_seperate_classifier:
                    d_loss_cls.backward()
                    self.dcls_optimizer.step()

                # Compute gradient penalty
                #if not self.onlymaskgen:
                if not (self.d_use_spectralnorm):
                    d_loss_gp = self.compute_gradient_penalty(real_x, fake_x)
                    # Backward + Optimize
                    d_loss = self.lambda_gp * d_loss_gp
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()
                    loss['D/loss_gp']   = loss['D/loss_gp']   * avgexp + (1-avgexp ) * d_loss_gp.data[0]   if 'D/loss_gp'   in loss else d_loss_gp.data[0]

                # Logging
                loss['D/loss_real'] = loss['D/loss_real'] * avgexp + (1-avgexp ) * d_loss_real.data[0] if 'D/loss_real' in loss else d_loss_real.data[0]
                loss['D/loss_fake'] = loss['D/loss_fake'] * avgexp + (1-avgexp ) * d_loss_fake.data[0] if 'D/loss_fake' in loss else d_loss_fake.data[0]
                loss['Wass_dist'] = loss['Wass_dist'] * avgexp + (1-avgexp ) * (-d_loss_real.data[0]-d_loss_fake.data[0]) if 'Wass_dist' in loss else (-d_loss_real.data[0]-d_loss_fake.data[0])
                loss['D/loss_cls']  = loss['D/loss_cls']  * avgexp + (1-avgexp ) * d_loss_cls.data[0]  if 'D/loss_cls'  in loss else d_loss_cls.data[0]
                #del grad, interpolated, out, out_cls, grad_l2norm, d_loss_gp, alpha
                #gc.collect()

                # ================== Train G ================== #
                if (i+1) % self.d_train_repeat == 0:
                    for p in self.G.parameters():
                        p.requires_grad = p.requires_grad_orig#

                    for p in self.D.parameters():
                        p.requires_grad = False#

                    if self.use_seperate_classifier:
                        for p in self.D_cls.parameters():
                            p.requires_grad = False#

                    # Original-to-target and Target-to-original domain
                    #if ('G/loss_rec' in loss) and (loss['G/loss_rec'] < 0.22):
                    #    import ipdb; ipdb.set_trace()

                    if self.dataset == 'coco':
                        fake_c = fake_label.clone()
                    fake_x, mask = self.G(real_x, fake_c, out_diff=True)

                    ## Should this be detached!?
                    if self.lambda_rec:
                        rec_x = self.G(fake_x, real_c)
                        if self.adv_rec:
                            out_src_rec, out_cls_rec = self.D(rec_x)

                    g_loss_vgg =0.
                    if self.lambda_vggloss:
                        g_loss_vgg = self.vggLoss(fake_x, real_x)
                        g_loss_vgg += (mask[1].mean())
                        loss['G/l_vgg'] = loss['G/l_vgg'] * avgexp + (1-avgexp ) * (self.lambda_vggloss * g_loss_vgg.data[0]) if 'G/l_vgg' in loss else (self.lambda_vggloss*g_loss_vgg.data[0])

                    g_smooth_loss = 0.
                    if self.lambda_smoothloss:
                        g_smooth_loss = F.conv2d(F.pad(mask[1],(1,1,1,1),mode='replicate'),smoothWeight).mean()
                        loss['G/l_sm'] = loss['G/l_sm'] * avgexp + (1-avgexp ) * (self.lambda_smoothloss * g_smooth_loss.data[0]) if 'G/l_sm' in loss else (self.lambda_smoothloss*g_smooth_loss.data[0])

                    # Compute losses
                    out_src, out_cls = self.D(fake_x)
                    if self.use_seperate_classifier:
                        out_cls = self.D_cls(fake_x)

                    if not self.onlymaskgen:
                        g_loss_fake = self.compute_real_fake_loss(out_src, self.adv_loss_type, loss_for = 'generator')
                        #if self.use_topk_patch == 0:
                        #    g_loss_fake = - torch.mean(out_src)
                        #else:
                        #    g_loss_fake = - out_src.view(cbsz,-1).topk(self.use_topk_patch,dim=1,largest=False)[0].mean()
                        loss['G/loss_fake'] = loss['G/loss_fake'] * avgexp + (1-avgexp ) * g_loss_fake.data[0] if 'G/loss_fake' in loss else g_loss_fake.data[0]
                    else:
                        g_loss_fake = 0.

                    if self.lambda_rec:
                        g_loss_rec = torch.mean(torch.abs(real_x - rec_x))
                        if self.adv_rec:
                            g_loss_rec_fake = - torch.mean(out_src_rec) if self.use_topk_patch == 0 else - out_src_rec.view(cbsz,-1).topk(self.use_topk_patch,dim=1,largest=False)[0].mean()
                            g_loss_rec_cls = F.binary_cross_entropy_with_logits(
                                               out_cls_rec.view(cbsz,-1), real_label, size_average=False) / cbsz
                            g_loss_rec += g_loss_rec_fake + g_loss_rec_cls
                    else:
                        g_loss_rec = 0.

                    if self.dataset in cocoAndCelebset:
                        g_loss_cls = F.binary_cross_entropy_with_logits(
                            out_cls.view(cbsz,-1), fake_label, size_average=False) / cbsz
                    else:
                        g_loss_cls = F.cross_entropy(out_cls, fake_label)

                    # Backward + Optimize
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + self.lambda_vggloss* g_loss_vgg + self.lambda_smoothloss * g_smooth_loss
                    # Add L1 loss to keep the difference image from going beyond boundaries
                    if self.differential_generator:
                        g_loss_abs = (Variable((torch.abs(fake_x.data)>1.0).float(),requires_grad=False)*(torch.abs(fake_x))).mean()
                        g_loss += g_loss_abs

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    if self.lambda_rec:
                        loss['G/loss_rec']  = loss['G/loss_rec']  * avgexp + (1-avgexp ) * g_loss_rec.data[0]  if 'G/loss_rec'  in loss else g_loss_rec.data[0]
                    loss['G/loss_cls']  = loss['G/loss_cls']  * avgexp + (1-avgexp ) * g_loss_cls.data[0]  if 'G/loss_cls'  in loss else g_loss_cls.data[0]

                    for p in self.D.parameters():
                        p.requires_grad = p.requires_grad_orig#

                    if self.use_seperate_classifier:
                        for p in self.D_cls.parameters():
                            p.requires_grad = True#


                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.2f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

                # Translate fixed images for debugging
                if (i+1) % self.sample_step == 0:
                    fake_image_list = [fixed_x]
                    for fixed_c in fixed_c_list:
                        fake_image_list.append(self.G(fixed_x, fixed_c))
                    fake_images = torch.cat(fake_image_list, dim=3)
                    save_image(self.denorm(fake_images.data),
                        os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
                    print('Translated images and saved into {}..!'.format(self.sample_path))

                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    checkpointData = {
                        'iter': e*iters_per_epoch+i,
                        'arch': self.arch,
                        'generator_state_dict':self.G.state_dict(),
                        'discriminator_state_dict': self.D.state_dict(),
                        'enc_dec_optimizer' : self.g_optimizer.state_dict(),
                        'discriminator_optimizer' : self.d_optimizer.state_dict(),
                    }
                    if self.use_seperate_classifier:
                        checkpointData['discriminator_cls_state_dict'] = self.D_cls.state_dict()
                        checkpointData['discriminator_cls_optimizer'] = self.dcls_optimizer.state_dict()

                    save_checkpoint(checkpointData, fappend = self.arch['fappend']+'_{}_{}.pth.tar'.format(e+1,i+1),
                        outdir = self.model_save_path)

                    #torch.save(self.G.state_dict(),
                    #    os.path.join(self.model_save_path, '{}_{}_G.pth'.format(e+1, i+1)))
                    #torch.save(self.D.state_dict(),
                    #    os.path.join(self.model_save_path, '{}_{}_D.pth'.format(e+1, i+1)))

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(10.))
                d_lr -= (self.d_lr / float(10.))
                e_lr -= (self.e_lr / float(10.))
                g_lr = max(1e-5, g_lr)
                d_lr = max(1e-5, d_lr)
                e_lr = max(1e-5, e_lr)
                self.update_lr(g_lr, d_lr, e_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def forward_fulleditor(self, x, label, get_feat=False, binary_mask=None, onlyMasks=False, mask_threshold=0.3,
                           gtMask=None, withGTMask=False, dilate=None, getAllMasks=False, n_iter = 0):

        # extract features from the box
        if withGTMask or self.only_gt_mask:
            mask = gtMask
            boxFeat = None
            allMasks = None
        else:
            binary_mask = (self.mode=='test' or self.mode=='eval') if binary_mask == None else binary_mask
            if self.use_gtmask_inp:
                e_inp = torch.cat([x,gtMask],dim=1)
            else:
                e_inp = x
            _, mask, boxFeat, allMasks = self.E(e_inp , label, binary_mask=binary_mask, mask_threshold = mask_threshold)
            if onlyMasks:
                return None, None, mask
        # add noise to this ?
        maskUpsamp = self.getImageSizeMask(mask)
        if dilate is not None:
            dsz = dilate.size(-1)//2
            maskUpsamp = torch.clamp(F.conv2d(F.pad(maskUpsamp,(dsz,dsz,dsz,dsz)),dilate), max=1.0, min=0.0)
            mask = maskUpsamp
        xM = (1-maskUpsamp)*x
        if self.e_addnoise:
            std = boxFeat.norm(dim=1)/25.
            boxFeat = boxFeat + torch.normal(means=0., std = std).view(-1,1).expand(boxFeat.size())

        # Pass the boxfeature and masked image to the
        if self.no_inpainter:
            fakeImg = xM
            feat = None
        else:
            #mask the input image and append the mask as a channel
            xInp =torch.cat([xM,maskUpsamp],dim=1)

            if get_feat:
                genImg, feat = self.G(xInp, boxFeat, out_diff=get_feat)
            else:
                genImg = self.G(xInp, boxFeat)
                feat = None

            if self.gen_fullimage:
                fakeImg = genImg
            else:
                fakeImg = genImg*maskUpsamp+ xM

            if n_iter > 0:
                for i in xrange(n_iter):
                    xInp =torch.cat([fakeImg,maskUpsamp],dim=1)
                    genImg = self.G(xInp, boxFeat)
                    fakeImg = genImg*maskUpsamp+ xM

        if getAllMasks:
            return fakeImg, feat, mask, allMasks
        else:
            return fakeImg, feat, mask

    def train_fulleditor(self):

        # Set dataloader
        self.data_loader = self.celebA_loader

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        fixed_x = []
        real_c = []
        fixed_gTMask = []
        cls_idx = []
        max_samples = 8
        for i, (images, labels, boxImg, boxlabel, mask, bbox, curCls) in enumerate(self.data_loader):
            fixed_x.append(images)
            real_c.append(labels)
            cls_idx.append(curCls[:,0])
            if self.use_gtmask_inp:
                fixed_gTMask.append(mask[:,1:,::])
            elif self.train_g_wo_m:
                fixed_gTMask.append(mask)
            if i == 0:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)[:max_samples]
        fixed_x = self.to_var(fixed_x, volatile=True)
        if self.use_gtmask_inp or self.train_g_wo_m:
            fixed_gTMask = torch.cat(fixed_gTMask, dim=0)[:max_samples]
            fixed_gTMask = self.to_var(fixed_gTMask, volatile=True)
        fixed_real_label = torch.cat(real_c, dim=0)[:max_samples]
        cls_idx = torch.cat(cls_idx, dim=0)[:max_samples]
        nnz_idx = fixed_real_label.sum(dim=1).nonzero().squeeze()
        if not self.train_g_wo_m:
            fixed_mask_target = torch.zeros_like(fixed_real_label)
            fixed_mask_target[nnz_idx.long(), cls_idx[nnz_idx.cpu()]] = 1.
            fixed_mask_target = self.to_var(fixed_mask_target,volatile=True)
        else:
            fixed_mask_target = None

        fixed_real_label = self.to_var(fixed_real_label,volatile=True)

        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr
        e_lr = self.e_lr

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[-2])
        else:
            start = 0

        if self.lambda_smoothloss:
            smoothWeight = -1.0/9.*torch.ones((1,1,3,3))
            smoothWeight[0,0,1,1] = 1.0
            smoothWeight = Variable(smoothWeight,requires_grad=False).cuda()

        if self.lambda_tvloss:
            tvWeight = torch.zeros((1,3,3,3))
            tvWeight[0,:,1,1] = -2.0
            tvWeight[0,:,1,2] = 1.0; tvWeight[0,:,2,1] = 1.0;
            tvWeight = Variable(tvWeight,requires_grad=False).cuda()
        # Start training
        loss = OrderedDict()
        accLog = {}
        avgexp= 0.95
        start_time = time.time()
        if self.use_maskprior_gan and not self.maskprior_matchclass:
            mask_loader = iter(self.mask_loader)

        print_iter = 1 if self.dataset == 'coco' or self.dataset=='places2'  or self.dataset=='ade20k' else 2
        old_mask = []
        if self.dilateMask:
            dilateWeight = torch.ones((1,1,self.dilateMask,self.dilateMask))
            dilateWeight = Variable(dilateWeight,requires_grad=False).cuda()
        else:
            dilateWeight = None


        for e in range(start, self.num_epochs):
            # Set some flags to decide what to train now.
            train_m = (not self.fixed_m) if self.alternate_mask_train < 2 else (not self.fixed_m) if (((e%self.train_g_every)!= 1) and self.alternate_mask_train==2) else False
            train_g = self.g_fine_tune if (self.alternate_mask_train< 2) or self.fixed_m  else self.g_fine_tune if (((e%self.train_g_every)==1) and self.alternate_mask_train==2) else 0
            if self.train_only_d:
                train_m = 1
                train_g = 0

            for i, (real_x, real_label, boxImg, boxlabel, randmask, bbox, curCls) in enumerate(self.data_loader):
                # Generat fake labels randomly (target domain labels)
                cbsz = real_label.size(0)
                nnz_idx = real_label.sum(dim=1).nonzero().squeeze().cuda()

                #Choose one of the classes for manipulation
                # Chose the class with label 1
                #rand_idx = torch.LongTensor([random.choice(real_label[bi,:].nonzero())[0] if len(real_label[bi,:].nonzero()) else 0 for bi in xrange(cbsz)])
                rand_idx = curCls[:,0]

                aidx= np.arange(cbsz)
                fake_label = real_label.clone()
                # XXX: FIX THIS!!!!!
                fake_label[aidx, rand_idx] =  0. #1. - real_label[aidx,rand_idx] if not self.arch['only_remove_train'] else 0.

                # This variable informs to the mask generator, which class to generate for
                mask_target = torch.zeros_like(real_label)
                if len(nnz_idx)>0:
                    mask_target[nnz_idx.cpu().long(), rand_idx[nnz_idx.cpu()]] = 1.

                for p in self.G.parameters():
                    p.requires_grad = False#
                if self.E is not None:
                    for p in self.E.parameters():
                        p.requires_grad = False#

                # Convert tensor to variable
                if self.use_gtmask_inp:
                    gtMask = randmask[:,1:,::]
                    randmask = randmask[:,:1,::]
                    gtMask = self.to_var(gtMask)
                else:
                    gtMask = None

                real_x = self.to_var(real_x)
                randmask = self.to_var(randmask)
                real_label = self.to_var(real_label)
                mask_target = self.to_var(mask_target)
                fake_label = self.to_var(fake_label)
                rand_idx = rand_idx.cuda()

                # ================== Train D ================== #

                #------------------------------------------------------------------------
                # TODO : Right now using all the images for the real/fake discriminator ?
                # Try using only non person images as real ?


                # Compute loss with fake images
                if self.only_random_boxes_discr:
                    fake_x = self.forward_boxreconst(real_x, None, randmask, no_enc = True)
                    fake_x = fake_x.detach()
                    if train_g and not self.train_g_wo_m:
                        fake_x_full, _, f_mask = self.forward_fulleditor(real_x, mask_target, binary_mask=False, gtMask = gtMask)
                        fake_x_full.detach()
                else:
                    fake_x, _, f_mask = self.forward_fulleditor(real_x, mask_target, binary_mask=False, gtMask = gtMask, dilate=dilateWeight)
                    fake_x = fake_x.detach()
                    f_mask = f_mask.detach()

                #------------------------------------------------------------------------
                # Compute loss with real images
                #------------------------------------------------------------------------
                if train_g:
                    rn = np.random.rand()< 0.5 if not self.train_g_wo_m else 1
                    dInp_real_x = real_x if not self.discrim_masked_image else torch.cat([real_x, real_x*(randmask if rn else self.getImageSizeMask(f_mask))], dim=1)
                    out_src, out_cls_real = self.D(dInp_real_x)
                    if self.d_local_supervision:
                        d_loss_real_p, d_loss_real_n, n_dlr_p, n_dlr_n = self.compute_real_fake_loss_local(out_src, self.adv_loss_type,
                                F.adaptive_max_pool2d((randmask if rn else self.getImageSizeMask(f_mask)),self.d_patch_size), only_pos = True)
                    else:
                        d_loss_real = self.compute_real_fake_loss(out_src, self.adv_loss_type, datasrc = 'real')
                    #if i > 200:
                    #  import ipdb; ipdb.set_trace()

                    dInp_fake_x = fake_x if not self.discrim_masked_image else torch.cat([fake_x, fake_x*(randmask if self.only_random_boxes_discr else self.getImageSizeMask(f_mask))], dim=1)
                    out_src_fake, out_cls_fake = self.D(dInp_fake_x)
                    if self.d_local_supervision:
                        d_loss_fake_p, d_loss_fake_n, n_dlf_p, n_dlf_n = self.compute_real_fake_loss_local(out_src_fake, self.adv_loss_type, F.adaptive_max_pool2d(randmask if self.only_random_boxes_discr else f_mask, self.d_patch_size))
                    else:
                        d_loss_fake = self.compute_real_fake_loss(out_src_fake, self.adv_loss_type, datasrc = 'fake')

                    # Backward + Optimize
                    if not self.only_random_boxes_discr:
                        rn = True
                        if not self.d_local_supervision:
                            d_loss = d_loss_real + d_loss_fake
                        else:
                            d_loss_real = (d_loss_real_p + d_loss_fake_p) / (1e-8 + n_dlr_p + n_dlf_p)
                            d_loss_fake = (d_loss_real_n + d_loss_fake_n) / (1e-8 + n_dlr_n + n_dlf_n)
                            d_loss = d_loss_real + d_loss_fake
                    else:
                        if self.train_g_wo_m:
                            if not self.d_local_supervision:
                                d_loss = 2.0*d_loss_real + d_loss_fake
                            else:
                                d_loss_real = (d_loss_real_p + d_loss_fake_p ) / (1e-8 + n_dlr_p + n_dlf_p)
                                d_loss_fake = (d_loss_real_n + d_loss_fake_n ) / (1e-8 + n_dlr_n + n_dlf_n)
                                d_loss = d_loss_real + d_loss_fake
                        else:
                            dInp_fake_x_full = fake_x_full if not self.discrim_masked_image else torch.cat([fake_x_full, fake_x_full*self.getImageSizeMask(f_mask)], dim=1)
                            out_src_fake_full, _= self.D(dInp_fake_x_full)
                            if self.d_local_supervision:
                                d_loss_fake_full_p, d_loss_fake_full_n, n_dlff_p, n_dlff_n = self.compute_real_fake_loss_local(out_src_fake_full, self.adv_loss_type, F.adaptive_max_pool2d(f_mask, self.d_patch_size))
                            else:
                                d_loss_fake_full = self.compute_real_fake_loss(out_src_fake_full, self.adv_loss_type, datasrc = 'fake')
                            if not self.d_local_supervision:
                                d_loss = 2.0*d_loss_real + d_loss_fake + d_loss_fake_full
                            else:
                                d_loss_real = (d_loss_real_p + d_loss_fake_p + d_loss_fake_full_p) / (1e-8 + n_dlr_p + n_dlf_p + n_dlff_p)
                                d_loss_fake = (d_loss_real_n + d_loss_fake_n + d_loss_fake_full_n) / (1e-8 + n_dlr_n + n_dlf_n + n_dlff_n)
                                d_loss = d_loss_real + d_loss_fake

                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()
                    # Compute gradient penalty
                    #if not self.onlymaskgen:
                    if not (self.d_use_spectralnorm) and self.lambda_gp > 0.:
                        d_loss_gp = self.compute_gradient_penalty(dInp_real_x, dInp_fake_x if rn else dInp_fake_x_full)

                        # Backward + Optimize
                        d_loss = self.lambda_gp * d_loss_gp
                        self.reset_grad()
                        d_loss.backward()
                        self.d_optimizer.step()
                        loss['D/l_gp']   = loss['D/l_gp']   * avgexp + (1-avgexp ) * d_loss_gp.data[0]   if 'D/l_gp'   in loss else d_loss_gp.data[0]
                    # Logging
                    loss['D/l_rf'] = loss['D/l_rf'] * avgexp + (1-avgexp ) * (d_loss_real+d_loss_fake).data[0] if 'D/l_rf' in loss else (d_loss_real+d_loss_fake).data[0]
                    #loss['D/l_real'] = loss['D/l_real'] * avgexp + (1-avgexp ) * d_loss_real.data[0] if 'D/l_real' in loss else d_loss_real.data[0]
                    #loss['D/l_fake'] = loss['D/l_fake'] * avgexp + (1-avgexp ) * d_loss_fake.data[0] if 'D/l_fake' in loss else d_loss_fake.data[0]
                    loss['Wass'] = loss['Wass'] * avgexp + (1-avgexp ) * (-d_loss_real.data[0]-d_loss_fake.data[0]) if 'Wass' in loss else (-d_loss_real.data[0]-d_loss_fake.data[0])


                if train_m and (not self.fixed_classifier):
                    if self.use_seperate_classifier:
                        _, out_cls_real, d_feat_real = self.D_cls(real_x, get_feat = True)
                    d_loss_cls = F.binary_cross_entropy_with_logits(
                        out_cls_real.view(cbsz, -1), np.random.uniform(0.95, 1.0)*real_label, size_average=False) / cbsz
                    if self.use_seperate_classifier and (self.adv_classifier or self.train_robust_d):
                        _, out_cls_fake, d_feat_fake = self.D_cls(fake_x, get_feat=True)

                    if self.adv_classifier:
                        # non flipped indices
                        nonFlipped_idx = (out_cls_fake[aidx,rand_idx] > -1).nonzero()
                        if len(nonFlipped_idx):
                            nonFlipped_idx = nonFlipped_idx.data[:,0]
                            factor = 0.2 if not self.only_random_boxes_discr else 1.0
                            d_loss_cls = d_loss_cls + factor*(F.binary_cross_entropy_with_logits(
                                out_cls_fake[nonFlipped_idx,:], np.random.uniform(0.95, 1.0)*real_label[nonFlipped_idx,:], size_average=False) / len(nonFlipped_idx))
                    elif self.train_robust_d:
                        factor = 1.0
                        d_loss_cls = d_loss_cls + factor*(F.binary_cross_entropy_with_logits(out_cls_fake.view(cbsz,-1), fake_label, size_average=False)/cbsz)
                        if self.compositional_loss>0.:
                            feat_mask = (1.-F.adaptive_max_pool2d(f_mask,d_feat_fake.shape[-1])).detach()
                            d_comp_loss = self.compositional_loss*F.mse_loss(d_feat_fake*feat_mask, d_feat_real*feat_mask)
                            d_loss_cls = d_loss_cls+d_comp_loss
                            loss['D/l_comp']  = loss['D/l_comp']  * avgexp + (1-avgexp ) * d_comp_loss.data[0]  if 'D/l_cls'  in loss else d_comp_loss.data[0]

                    # Compute classification accuracy of the discriminator
                    if (i+1) % self.log_step == 0:
                        r_accuracies = self.compute_accuracy(out_cls_real.view(cbsz,-1), real_label, self.dataset)
                        accLog['r_acc'] = accLog['r_acc'] * 0.8 + (1-0.8)* r_accuracies.data.cpu().numpy() if 'r_acc' in accLog else  r_accuracies.data.cpu().numpy()
                        if (not self.only_random_boxes_discr) and (self.adv_classifier or self.train_robust_d):
                            f_accuracies = self.compute_accuracy(out_cls_fake.view(cbsz,-1), real_label, self.dataset)
                            accLog['f_acc'] = accLog['f_acc'] * 0.8 + (1-0.8)* f_accuracies.data.cpu().numpy() if 'f_acc' in accLog else  f_accuracies.data.cpu().numpy()

                    if self.use_seperate_classifier:
                        self.reset_grad()
                        d_loss_cls.backward()
                        self.dcls_optimizer.step()
                    loss['D/l_cls']  = loss['D/l_cls']  * avgexp + (1-avgexp ) * d_loss_cls.data[0]  if 'D/l_cls'  in loss else d_loss_cls.data[0]

                #------------------------------------------------
                # Using mask discriminator to impose mask priors
                #------------------------------------------------
                if self.use_maskprior_gan and len(nnz_idx)>0 and (not self.fixed_m) and train_m and (not self.train_only_d):
                    if self.only_random_boxes_discr:
                        _, _, f_mask = self.forward_fulleditor(real_x, mask_target, binary_mask=False, onlyMasks=True, gtMask = gtMask)
                        f_mask.detach()
                    f_mask = f_mask[nnz_idx, ::].detach()
                    fm_targ = mask_target[nnz_idx,:]
                    # get a batch or real masks
                    # Right now the model is not using labels, but should change this soon.
                    if self.boxprior:
                        real_mask = F.adaptive_max_pool2d(randmask[:len(nnz_idx),::],self.mask_size).detach()
                        real_mlabel = fm_targ.clone()
                    else:
                        if not self.maskprior_matchclass:
                            try:
                                real_mask, real_mlabel = next(mask_loader)
                            except:
                                mask_loader = iter(self.mask_loader)
                                real_mask, real_mlabel = next(mask_loader)
                            real_mlabel = self.to_var(real_mlabel[:len(nnz_idx),:])
                            real_mask = self.to_var(real_mask[:len(nnz_idx),::])
                        else:
                            real_mask = self.mask_loader.dataset.getbyClass(fm_targ.data.nonzero()[:,1])
                            real_mlabel = fm_targ.clone()
                            real_mask = self.to_var(real_mask)
                    #real_mask = torch.clamp((real_mask*(torch.randn(*real_mask.size())*0.05) + real_mask), 0., 1.)
                    # Add instance noise:
                    f_mask = f_mask[:real_mask.size(0),::]
                    fm_targ = fm_targ[:real_mask.size(0),:]

                    rf_realmask,_ = self.mask_D(real_mask, label=real_mlabel)
                    rf_fakemask,_ = self.mask_D(f_mask, label=fm_targ)
                    md_loss_r = self.compute_real_fake_loss(rf_realmask, self.m_adv_loss_type, datasrc = 'real')
                    md_loss_f = self.compute_real_fake_loss(rf_fakemask, self.m_adv_loss_type, datasrc = 'fake')
                    md_loss = md_loss_r + md_loss_f
                    self.reset_grad()
                    md_loss.backward()
                    self.mask_d_optimizer.step()
                    if not (self.d_use_spectralnorm):
                        mask_d_loss_gp = self.lambda_gp * self.compute_gradient_penalty(real_mask, f_mask, discrim_type='mask', label=(real_mlabel+fm_targ)/2.)
                        self.reset_grad()
                        mask_d_loss_gp.backward()
                        self.mask_d_optimizer.step()
                    md_loss += mask_d_loss_gp
                    if self.use_maskprior_gan and len(nnz_idx)>0 and (not self.fixed_m):
                        #loss['MD/l_rf'] = loss['MD/l_rf'] * avgexp + (1-avgexp ) * md_loss.data[0] if 'MD/l_rf' in loss else md_loss.data[0]
                        loss['Wass_m'] = loss['Wass_m'] * avgexp + (1-avgexp ) * (-md_loss_r.data[0]-md_loss_f.data[0]) if 'Wass_m' in loss else (-md_loss_r.data[0]-md_loss_f.data[0])


                #del grad, interpolated, out, out_cls, grad_l2norm, d_loss_gp, alpha
                #gc.collect()

                # ================== Train G ================== #
                if (i+1) % self.d_train_repeat == 0 and (not self.train_only_d):
                    for p in self.G.parameters():
                        p.requires_grad = p.requires_grad_orig if (self.alternate_mask_train==2 and train_g) or (not self.alternate_mask_train) else False#

                    for p in self.D.parameters():
                        p.requires_grad = False#
                    if self.use_maskprior_gan:
                        for p in self.mask_D.parameters():
                            p.requires_grad = False#
                    if self.E is not None:
                        for p in self.E.parameters():
                            p.requires_grad =  p.requires_grad_orig if train_m else False

                    if (self.D_cls is not None) and self.use_seperate_classifier:
                        for p in self.D_cls.parameters():
                            p.requires_grad = False#


                    # Original-to-target and Target-to-original domain
                    #if ('G/loss_rec' in loss) and (loss['G/loss_rec'] < 0.22):
                    #    import ipdb; ipdb.set_trace()

                    if self.train_g_wo_m:
                        reasonable_masks = []
                        fake_x = []
                    else:
                        fake_x,_,mask, allMasks = self.forward_fulleditor(real_x, mask_target, binary_mask=not train_m, gtMask = gtMask, getAllMasks = True)
                        if train_g:
                            # Keep only reasonably sized masks:
                            reasonable_masks = (mask.data.mean(dim=2).mean(dim=2)[:,0]<0.5).nonzero()
                            if len(reasonable_masks):
                                reasonable_masks =reasonable_masks[:,0]
                                fake_x = fake_x[reasonable_masks,::]
                                mask = mask[reasonable_masks,::]
                            else:
                                fake_x = []

                    ## Should this be detached!?
                    #if self.lambda_rec:
                    #    rec_x = self.G(fake_x, real_c)
                    #    if self.adv_rec:
                    #        out_src_rec, out_cls_rec = self.D(rec_x)

                    #g_loss_vgg =0.
                    #if self.lambda_vggloss:
                    #    g_loss_vgg = self.vggLoss(fake_x, real_x)
                    #    loss['G/l_vgg'] = loss['G/l_vgg'] * avgexp + (1-avgexp ) * (self.lambda_vggloss * g_loss_vgg.data[0]) if 'G/l_vgg' in loss else (self.lambda_vggloss*g_loss_vgg.data[0])

                    if train_m:
                        m_l1_loss = 0.
                        if self.lambda_maskL1loss and not self.grad_weighted_l1loss:
                            hingeLoss = torch.clamp(mask-0.1, min=0.).mean(dim=2).mean(dim=2) #(mask>0.1).float().mean(dim=2).mean(dim=2)
                            m_l1_loss = torch.exp(hingeLoss).mean()
                            #m_l1_loss = hingeLoss.mean()
                            loss['M/l_l1'] = loss['M/l_l1'] * avgexp + (1-avgexp ) * (self.lambda_maskL1loss * m_l1_loss.data[0]) if 'M/l_sm' in loss else (self.lambda_maskL1loss*m_l1_loss.data[0])
                        m_smooth_loss = 0.
                        if self.lambda_smoothloss:
                            # Smoothing loss is a combination of l1 loss and
                            m_smooth_loss = F.conv2d(F.pad(mask,(1,1,1,1),mode='replicate'),smoothWeight).mean()
                            loss['M/l_sm'] = loss['M/l_sm'] * avgexp + (1-avgexp ) * (self.lambda_smoothloss * m_smooth_loss.data[0]) if 'M/l_sm' in loss else (self.lambda_smoothloss*m_smooth_loss.data[0])

                        if self.use_maskprior_gan and len(nnz_idx)>0:
                            mask_rf,_ = self.mask_D(mask[nnz_idx,:], mask_target[nnz_idx,:])
                            mask_loss_fake = self.compute_real_fake_loss(mask_rf, self.m_adv_loss_type, loss_for = 'generator')
                            #loss['M/l_fake'] = loss['M/l_fake'] * avgexp + (1-avgexp ) * mask_loss_fake.data[0] if 'M/l_fake' in loss else mask_loss_fake.data[0]
                        else:
                            mask_loss_fake = 0.

                    # Compute real/ fake losses
                    if len(fake_x):
                        #XXX: REMOVE THIS
                        #if train_g:
                        dInp_fake_x = fake_x if not self.discrim_masked_image else torch.cat([fake_x, fake_x*self.getImageSizeMask(mask)], dim=1)
                        out_src, out_cls = self.D(dInp_fake_x)
                        if (self.D_cls is not None) and self.use_seperate_classifier:
                            _, out_cls = self.D_cls(fake_x)

                    if self.only_random_boxes_discr and train_m:
                        if (i+1) % self.log_step == 0:
                            f_accuracies = self.compute_accuracy(out_cls.view(cbsz,-1), real_label, self.dataset)
                            accLog['f_acc'] = accLog['f_acc'] * 0.8 + (1-0.8)* f_accuracies.data.cpu().numpy() if 'f_acc' in accLog else  f_accuracies.data.cpu().numpy()

                    if train_m:
                        if self.useclsweights:
                            cls_scale = self.clsweight_scale_init**(e-start) if self.clsweight_scale_decay else self.clsweight_scale_init
                            weights = F.sigmoid(cls_scale*out_cls_real.detach().view(cbsz,-1)[nnz_idx, rand_idx[nnz_idx]])
                        else:
                            weights = None
                        if self.onlytargetclsloss==1:
                            if len(nnz_idx)>0:
                                g_loss_cls = F.binary_cross_entropy_with_logits(
                                        out_cls.view(cbsz,-1)[nnz_idx, rand_idx[nnz_idx]],fake_label[nnz_idx,rand_idx[nnz_idx]], size_average=False, weight=weights) / len(nnz_idx)
                            else:
                                g_loss_cls = 0.
                        elif self.onlytargetclsloss==2:
                            g_loss_cls = 0.
                            if len(nnz_idx):
                                # For objects to be removed, classifier should predict zero label
                                scores_remove = out_cls[nnz_idx,rand_idx[nnz_idx]]
                                g_loss_cls = 1.2*F.binary_cross_entropy_with_logits(scores_remove, torch.zeros_like(scores_remove).detach(), weight=weights)
                                #g_loss_cls = torch.pow(torch.clamp(scores_remove+5., min=0.),1.5).mean()

                                # For other present objects, classifier scores shouldn't decrease below the score for real label
                                scores_retain = out_cls[:][fake_label.data[:].byte()]
                                scores_retain_real = out_cls_real[:][fake_label.data[:].byte()].detach()
                                scores_retain_damaged = scores_retain[(scores_retain_real > scores_retain).detach()]
                                if self.useclsweights:
                                    weights = F.sigmoid(cls_scale*scores_retain_real[(scores_retain_real > scores_retain).detach()])
                                else:
                                    weights = None

                                #g_loss_cls += torch.pow(torch.clamp(torch.clamp(scores_retain_real,max=3.) - scores_retain,min=0.),1.5).mean()
                                g_loss_cls += F.binary_cross_entropy_with_logits(scores_retain_damaged, torch.ones_like(scores_retain_damaged).detach(), weight=weights)
                        else:
                            g_loss_cls = F.binary_cross_entropy_with_logits(out_cls.view(cbsz,-1), fake_label, size_average=False) / cbsz

                        if self.mask_normalize_byclass:
                            # Here we maximize the probability for labels currently present (including background), minimize for absent classes
                            bsz, nC = allMasks.size(0), allMasks.size(1) -1
                            hyppar_r = 5.
                            present_classes = torch.cat([real_label,torch.ones_like(real_label[:,:1])],dim=1)

                            #-------------------------------------------------------------------------------------------------
                            # Loss Function is (maximize present classes), minimize the absent ones
                            # "allMaks" alread has normalized scores against all classes for each pixel location
                            # Now we need to decide how to aggreagte these each pixel scores into per-class score, since we only have class level GT label
                            # Simplest solution is to use mean/ max. But here we use the log-sum-exp to get a trade-off b/w mean and max extremes.
                            #-------------------------------------------------------------------------------------------------

                            flat_scores = allMasks[:,:nC,::].contiguous().view(bsz,nC,-1)
                            #lse_scores = torch.log(torch.exp(hyppar_r*flat_scores).mean(dim=-1))/hyppar_r
                            lse_scores,_ = torch.max(flat_scores, dim=-1)

                            # present_classes^1 performs negation.
                            m_loss_class = F.binary_cross_entropy(lse_scores, present_classes, size_average=False)/float(bsz) #-torch.log(lse_scores[present_classes]).mean() - torch.log(1.-lse_scores[present_classes^1]).mean()
                            loss['M/l_cls']  = loss['M/l_cls']  * avgexp + (1-avgexp ) * m_loss_class.data[0]  if 'M/l_cls'  in loss else m_loss_class.data[0]
                        else:
                            m_loss_class = 0.

                        loss['G/l_cls']  = loss['G/l_cls']  * avgexp + (1-avgexp ) * g_loss_cls.data[0]  if 'G/l_cls'  in loss else g_loss_cls.data[0]


                    if len(fake_x) and train_g:
                        if self.d_local_supervision:
                            g_loss_fake,_,n_glf_p,_ = self.compute_real_fake_loss_local(out_src, self.adv_loss_type, F.adaptive_max_pool2d(mask, self.d_patch_size), only_pos = True)
                            g_loss_fake = g_loss_fake / (1e-8+n_glf_p)
                        else:
                            g_loss_fake = self.compute_real_fake_loss(out_src, self.adv_loss_type, loss_for = 'generator')
                    else:
                        #XXX: REMOVE THIS!!!!
                        if self.d_local_supervision:
                            g_loss_fake,_,n_glf_p,_ = self.compute_real_fake_loss_local(out_src, self.adv_loss_type, F.adaptive_max_pool2d(mask, self.d_patch_size), only_pos = True)
                            g_loss_fake = g_loss_fake / (1e-8+n_glf_p)
                        else:
                            g_loss_fake = self.compute_real_fake_loss(out_src, self.adv_loss_type, loss_for = 'generator')
                        #g_loss_fake = 0.

                    if train_g:
                        if len(reasonable_masks):
                            g_loss_rec = 0. #F.smooth_l1_loss(fake_x,real_x[reasonable_masks,::]) if self.lambda_rec and not self.only_random_boxes_discr else 0.
                            g_loss_vgg = 0. #self.vggLoss(fake_x, real_x[reasonable_masks,::]) if self.lambda_vggloss and not self.only_random_boxes_discr else 0.
                            #if self.lambda_rec and not self.only_random_boxes_discr:
                            #    loss['G/l_rec']  = loss['G/l_rec']  * avgexp + (1-avgexp ) * g_loss_rec.data[0]  if 'G/l_rec'  in loss else g_loss_rec.data[0]

                    if train_m and train_g:
                        gm_loss = g_loss_fake+ self.lambda_cls * g_loss_cls + self.lambda_smoothloss * m_smooth_loss + self.lambda_maskfake_loss * mask_loss_fake + self.lambda_maskL1loss * m_l1_los + m_loss_classs
                        self.reset_grad()
                        gm_loss.backward()
                        self.g_optimizer.step()
                        self.e_optimizer.step()
                    elif train_m:
                        self.reset_grad()
                        if self.lambda_maskL1loss and self.grad_weighted_l1loss:
                            cls_loss = self.lambda_cls * g_loss_cls
                            mask.retain_grad()
                            cls_loss.backward(retain_graph=True, create_graph=True)

                            grad_weight = 1.-(mask.grad.abs()/mask.grad.abs().view(mask.size(0),-1).max(dim=1)[0])
                            hingeLoss = torch.clamp(mask*grad_weight-0.1, min=0.).mean(dim=2).mean(dim=2) #(mask>0.1).float().mean(dim=2).mean(dim=2)
                            m_l1_loss = torch.exp(hingeLoss).mean()
                            loss['M/l_l1'] = loss['M/l_l1'] * avgexp + (1-avgexp ) * (self.lambda_maskL1loss * m_l1_loss.data[0]) if 'M/l_sm' in loss else (self.lambda_maskL1loss*m_l1_loss.data[0])
                            m_loss = g_loss_fake+ self.lambda_smoothloss * m_smooth_loss + self.lambda_maskfake_loss * mask_loss_fake + self.lambda_maskL1loss * m_l1_loss + m_loss_class
                        else:
                            m_loss = g_loss_fake + self.lambda_cls * g_loss_cls + self.lambda_smoothloss * m_smooth_loss + self.lambda_maskfake_loss * mask_loss_fake + self.lambda_maskL1loss * m_l1_loss + m_loss_class
                            loss['G/l_fak']  = loss['G/l_fak']  * avgexp + (1-avgexp ) * g_loss_fake.data[0]  if 'G/l_fk'  in loss else g_loss_fake.data[0]
                        m_loss.backward()
                        self.e_optimizer.step()
                    elif train_g:
                        self.reset_grad()
                        # Backward + Optimize
                        if len(reasonable_masks):
                            if self.lambda_tvloss:
                                # Smoothing loss is a combination of l1 loss and
                                g_tv_loss = (torch.abs(F.conv2d(F.pad(fake_x,(1,1,1,1),mode='replicate'),tvWeight))).mean()
                                loss['G/l_tv'] = loss['G/l_tv'] * avgexp + (1-avgexp ) * (self.lambda_tvloss* g_tv_loss.data[0]) if 'G/l_tv' in loss else (self.lambda_tvloss*g_tv_loss.data[0])
                            else:
                                g_tv_loss  = 0.
                            g_loss = 2*g_loss_fake +self.lambda_rec * g_loss_rec + self.lambda_vggloss * g_loss_vgg  + self.lambda_tvloss * g_tv_loss
                            g_loss.backward()
                            #(g_loss_fake+ 0.1*g_loss_cls).backward()
                            #(g_loss_fake).backward()
                            #g_loss.backward()
                            self.g_optimizer.step()

                        if self.use_random_boxes:
                            if self.use_past_masks and (np.random.rand() < 0.5) and len(old_mask):
                                real_x = real_x[:len(old_mask)]
                                randmask = self.getImageSizeMask(old_mask[:real_x.size(0)])
                                fake_x = self.forward_boxreconst(real_x, None, randmask, no_enc = True)
                            else:
                                fake_x = self.forward_boxreconst(real_x, None, randmask, no_enc = True)
                            #g_loss_rec = F.smooth_l1_loss(fake_x,real_x) if self.lambda_rec else 0.
                            g_loss_rec = F.l1_loss(fake_x,real_x) if self.lambda_rec else 0.
                            if self.lambda_tvloss:
                                # Smoothing loss is a combination of l1 loss and
                                g_tv_loss = (torch.abs(F.conv2d(F.pad(fake_x,(1,1,1,1),mode='replicate'),tvWeight))).mean()
                                loss['G/l_tv'] = loss['G/l_tv'] * avgexp + (1-avgexp ) * (self.lambda_tvloss* g_tv_loss.data[0]) if 'G/l_tv' in loss else (self.lambda_tvloss*g_tv_loss.data[0])
                            else:
                                g_tv_loss  = 0.

                            g_loss_vgg = self.vggLoss(fake_x, real_x) if self.lambda_vggloss else 0.
                            if self.lambda_vggloss:
                                loss['G/l_vgg'] = loss['G/l_vgg'] * avgexp + (1-avgexp ) * (self.lambda_vggloss * g_loss_vgg.data[0]) if 'G/l_vgg' in loss else (self.lambda_vggloss*g_loss_vgg.data[0])
                            if self.lambda_rec:
                                loss['G/l_rec']  = loss['G/l_rec']  * avgexp + (1-avgexp ) * self.lambda_rec*g_loss_rec.data[0]  if 'G/l_rec'  in loss else self.lambda_rec*g_loss_rec.data[0]
                            dInp_fake_x = fake_x if not self.discrim_masked_image else torch.cat([fake_x, fake_x*randmask], dim=1)
                            out_src, _ = self.D(dInp_fake_x)
                            if self.d_local_supervision:
                                g_loss_fake,_, n_glf_p, _ = self.compute_real_fake_loss_local(out_src, self.adv_loss_type, F.adaptive_max_pool2d(randmask, self.d_patch_size), only_pos = True)
                                g_loss_fake = g_loss_fake / (1e-8+n_glf_p)
                            else:
                                g_loss_fake = self.compute_real_fake_loss(out_src, self.adv_loss_type, loss_for = 'generator')
                            self.reset_grad()
                            # Backward + Optimize
                            g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_vggloss * g_loss_vgg + self.lambda_tvloss * g_tv_loss
                            #if i > 1000:
                            #    import ipdb; ipdb.set_trace()
                            g_loss.backward()
                            self.g_optimizer.step()

                            if self.use_past_masks and len(reasonable_masks):
                                old_mask = mask.detach()


                    # Logging
                    #if self.lambda_rec:
                    #    loss['G/l_rec']  = loss['G/l_rec']  * avgexp + (1-avgexp ) * g_loss_rec.data[0]  if 'G/l_rec'  in loss else g_loss_rec.data[0]
                    for p in self.D.parameters():
                        p.requires_grad = p.requires_grad_orig#

                    if (self.D_cls is not None) and self.use_seperate_classifier:
                        for p in self.D_cls.parameters():
                            p.requires_grad = p.requires_grad_orig #

                    if self.use_maskprior_gan:
                        for p in self.mask_D.parameters():
                            p.requires_grad = p.requires_grad_orig#


                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]

                    log = "| T={} | E={}/{} | I={}/{}| - ".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += "|{}: {:.2f} ".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)
                #if i > 200:
                #    break

            if (e+1) % print_iter == 0:
                if 'r_acc' in accLog:
                    log = ['Real--']+["{}: {:.2f}".format(self.selected_attrs[cati],acc) for cati,acc in enumerate(accLog['r_acc'])]
                    print(' '.join(log))
                if 'f_acc' in accLog:
                    log = ['Fake--']
                    log.extend(["{}: {:.2f}".format(self.selected_attrs[cati],acc) for cati,acc in enumerate(accLog['f_acc'])])
                    print(' '.join(log))

            # Translate fixed images for debugging
            if ((e+1) % self.sample_step == 0) and (not self.train_only_d):
                fake_image_list = [fixed_x]
                fake_x,_,mask = self.forward_fulleditor(fixed_x, fixed_mask_target, binary_mask=True, gtMask = fixed_gTMask)
                mask = self.getImageSizeMask(mask)
                fake_image_list.append(mask.expand(*fixed_x.size()))
                fake_image_list.append(fake_x)
                fake_images = torch.cat(fake_image_list, dim=3)
                save_image(self.denorm(fake_images.data),
                    os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
                print('Translated images and saved into {} !'.format(self.sample_path))

                del fake_x, mask, fake_images, fake_image_list

            # Save model checkpoints
            if (e+1) % self.model_save_step == 0:
                if not self.train_only_d:
                    checkpointData = {
                        'iter': e*iters_per_epoch+i,
                        'arch': self.arch,
                        'generator_state_dict':self.G.state_dict(),
                        'discriminator_state_dict': self.D.state_dict(),
                    }
                else:
                    checkpointData = {
                        'iter': e*iters_per_epoch+i,
                        'arch': self.arch,
                    }
                if self.E is not None:
                    checkpointData['encoder_state_dict'] = self.E.state_dict()

                if (self.D_cls is not None) and self.use_seperate_classifier and not self.fixed_m:
                    checkpointData['discriminator_cls_state_dict'] = self.D_cls.state_dict()
                if self.use_maskprior_gan and (not (self.train_only_d or self.fixed_m)):
                    checkpointData['mask_discriminator_state_dict'] = self.mask_D.state_dict()

                save_checkpoint(checkpointData, fappend = self.arch['fappend']+'_{}_{}.pth.tar'.format(e+1,i+1),
                    outdir = self.model_save_path)

                #torch.save(self.G.state_dict(),
                #    os.path.join(self.model_save_path, '{}_{}_G.pth'.format(e+1, i+1)))
                #torch.save(self.D.state_dict(),
                #    os.path.join(self.model_save_path, '{}_{}_D.pth'.format(e+1, i+1)))

            # Decay learning rate
            if ((e+1) > (self.num_epochs - self.num_epochs_decay))  and e%self.decay_every == 0:
                g_lr -= (self.g_lr / float(10.))
                d_lr -= (self.d_lr / float(10.))
                e_lr -= (self.e_lr / float(10.))
                g_lr = max(1e-6, g_lr)
                d_lr = max(1e-5, d_lr)
                e_lr = max(1e-5, e_lr)
                self.update_lr(g_lr, d_lr, e_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def forward_boxreconst(self, x, box, mask, boxlabel=None, bbox=None, get_feat=False, no_enc = False):
        #mask the input image and append the mask as a channel
        xM = x*(1-mask)
        xInp =torch.cat([xM,mask],dim=1)

        # extract features from the box
        if self.E is not None and (not no_enc):
            if self.train_boxreconst == 2 or self.full_image_encoder:
                boxFeat = self.E(xInp if self.e_masked_image==1 else torch.cat([x, xInp],dim=1) if self.e_masked_image==2 else torch.cat([x, mask],dim=1) , boxlabel if self.use_box_label else None)
            else:
                boxFeat = self.E(box, boxlabel if self.use_box_label else None)
            # add noise to this ?
            if self.e_addnoise:
                std = boxFeat.norm(dim=1)/25.
                boxFeat = boxFeat + torch.normal(means=0., std = std).view(-1,1).expand(boxFeat.size())
        else:
            boxFeat = None

        # Pass the boxfeature and masked image to the
        if get_feat:
            genImg, feat = self.G(xInp, boxFeat, out_diff=get_feat)
        else:
            genImg = self.G(xInp, boxFeat)

        if self.g_fixed_size:
            bsz = x.size(0)
            imsz = x.size()[1:]
            fakeImg = xM
            for i in xrange(bsz):
                #r_genImg = [F.adaptive_avg_pool2d(genImg[i,:,:,:], (bbox[i,3],bbox[i,2])) for i in xrange(bsz)]
                fakeImg[i,:,bbox[i,1]:bbox[i,1]+bbox[i,3], bbox[i,0]:bbox[i,0]+bbox[i,2]] =  F.adaptive_avg_pool2d(genImg[i:i+1,:,:,:], (bbox[i,3],bbox[i,2]))
                #xM.masked_scatter(mask[i,::].byte().expand(*imsz), r_genImg[i])
        else:
            fakeImg = genImg*mask+ xM if not self.gen_fullimage else genImg
        if get_feat:
            return fakeImg, feat
        else:
            return fakeImg

    def getRandBoxWith(self, img, boxestoInc=None):
        imsz = img.size(-1)
        bsz = img.size(0)
        size = self.box_size
        if boxestoInc is None:
            randomBoxes = [(random.randint(0,imsz-size), random.randint(0,imsz-size)) for i in xrange(bsz)]
        else:
            # sample the new box such that the center of the target box is within it
            #randomBoxes = [(random.randint(min(max(boxestoInc[i,1]+boxestoInc[i,3]//2-(3*size//4), 0), imsz-size), min(max(boxestoInc[i,1]+boxestoInc[i,3]//2-(size//4),0), imsz-size)),
            #                random.randint(min(max(boxestoInc[i,0]+boxestoInc[i,2]//2-(3*size//4), 0), imsz-size), min(max(boxestoInc[i,0]+boxestoInc[i,2]//2-(size//4),0), imsz-size))) for i in xrange(bsz)]
            randomBoxes = [(random.randint(min(max(min(boxestoInc[i,1]+boxestoInc[i,3]-size, boxestoInc[i,1]), 0), imsz-size),
                                           min(max(max(boxestoInc[i,1]+boxestoInc[i,3]-size, boxestoInc[i,1]),0), imsz-size)),
                            random.randint(min(max(min(boxestoInc[i,0]+boxestoInc[i,2]-size, boxestoInc[i,0]), 0), imsz-size),
                                           min(max(max(boxestoInc[i,0]+boxestoInc[i,2]-size, boxestoInc[i,0]),0), imsz-size))) for i in xrange(bsz)]

        croppedImg = torch.cat([img[i:i+1, :, rb[0]:rb[0]+size, rb[1]:rb[1]+size] for i,rb in enumerate(randomBoxes)],dim=0)

        return croppedImg


    def train_reconst_nw(self):

        # Set dataloader
        self.data_loader = self.celebA_loader

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        fixed_x = []
        real_c = []
        fixed_box = []
        fixed_mask= []
        fixed_boxlabel = []
        fixed_bbox = []
        for i, (images, labels, boxImg, boxlabel, mask, bbox) in enumerate(self.data_loader):
            fixed_x.append(images)
            fixed_box.append(boxImg)
            fixed_mask.append(mask)
            fixed_boxlabel.append(boxlabel)
            fixed_bbox.append(bbox)
            if i == 3:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)
        fixed_box = torch.cat(fixed_box, dim=0)
        fixed_box = self.to_var(fixed_box, volatile=True)
        fixed_mask = torch.cat(fixed_mask, dim=0)
        fixed_mask = self.to_var(fixed_mask, volatile=True)
        fixed_boxlabel = torch.cat(fixed_boxlabel, dim=0)
        if self.train_boxreconst > 1:
            fixed_fake_boxlabel = torch.zeros_like(fixed_boxlabel)
        else:
            fixed_fake_boxlabel = fixed_boxlabel

        if self.use_box_label == 2:
            fixed_fake_boxlabel = torch.cat([fixed_boxlabel, fixed_fake_boxlabel],dim=1)

        fixed_boxlabel = self.to_var(fixed_fake_boxlabel, volatile=True)
        fixed_bbox = torch.cat(fixed_bbox, dim=0)

        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr
        e_lr = self.e_lr

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[-2])
        else:
            start = 0

        train_edit_mode = self.train_boxreconst > 1

        # Start training
        loss = {}
        accLog = {}
        avgexp= 0.95
        start_time = time.time()
        cocoAndCelebset= set(['CelebA', 'coco', 'mnist', 'celebbox', 'pascal'])
        for e in range(start, self.num_epochs):
            for i, (real_x, real_label, boxImg, boxlabel, mask, bbox) in enumerate(self.data_loader):
                # Generat fake labels randomly (target domain labels)
                #plt.imshow(((real_x.numpy()[6,[0,1,2],:,:].transpose(1,2,0)+1.0)*255./2.0).astype(np.uint8)); plt.show()
                cbsz = real_label.size(0)

                nnz_idx = boxlabel[:,0].nonzero().squeeze().cuda()
                z_idx = (1-boxlabel[:,0]).nonzero().squeeze().cuda()

                rand_idx = np.random.randint(0,boxlabel.size(1),cbsz)
                aidx= np.arange(cbsz)
                fake_boxlabel = boxlabel.clone()
                # XXX: FIX THIS!!!!!
                fake_boxlabel[aidx,rand_idx] =  (1. - boxlabel[aidx,rand_idx]) if not self.arch['only_remove_train'] else 0.
                if self.use_box_label ==2:
                    # Append the gt and target box label
                    fake_boxlabel_inp = torch.cat([boxlabel, fake_boxlabel], dim=1)
                    fake_boxlabel_inp_rec = torch.cat([fake_boxlabel, boxlabel], dim=1)
                    fake_boxlabel_inp_rec = self.to_var(fake_boxlabel_inp_rec)
                else:
                    fake_boxlabel_inp = fake_boxlabel


                #for p in self.G.parameters():
                #    p.requires_grad = False#
                #for p in self.E.parameters():
                #    p.requires_grad = False#

                # Convert tensor to variable
                real_x = self.to_var(real_x)
                boxImg = self.to_var(boxImg)
                boxlabel = self.to_var(boxlabel)
                fake_boxlabel = self.to_var(fake_boxlabel)
                fake_boxlabel_inp = self.to_var(fake_boxlabel_inp)
                mask = self.to_var(mask)

                # ================== Train D ================== #

                if not self.only_reconst_loss:
                    # Compute loss with real images
                    out_src, out_cls,_ = self.D(real_x, self.getRandBoxWith(real_x, boxestoInc=bbox), classify=train_edit_mode)

                    if train_edit_mode:
                        # in this case just do real/fake classification on images not containing person.
                        # TODO: This doesn't make sense
                        out_src_sel = out_src #[z_idx]
                    else:
                        out_src_sel = out_src

                    d_loss_real = self.compute_real_fake_loss(out_src, self.adv_loss_type,datasrc = 'real')

                    d_loss_cls = 0

                    if train_edit_mode:
                        d_loss_cls = F.binary_cross_entropy_with_logits(
                            out_cls.view(cbsz, -1), boxlabel, size_average=False)/ cbsz
                    #else:
                    #    d_loss_cls = F.cross_entropy(out_cls, boxlabel)

                    ## Compute classification accuracy of the discriminator
                    if train_edit_mode and ((i+1) % self.log_step == 0):
                        accuracies = self.compute_accuracy(out_cls.view(cbsz,-1), boxlabel, self.dataset)
                        accLog['acc'] = accLog['acc'] * 0.8 + (1-0.8)* accuracies.data.cpu().numpy() if 'acc' in accLog else  accuracies.data.cpu().numpy()
                        log = ["{}: {:.2f}".format(self.selected_attrs[cati],acc) for cati,acc in enumerate(accLog['acc'])]
                        print(log)

                    # Compute loss with fake images
                    fake_x_out = self.forward_boxreconst(real_x, boxImg, mask, fake_boxlabel_inp, bbox)
                    fake_x = Variable(fake_x_out.data)
                    out_src_fake, out_cls_fake, _ = self.D(fake_x, self.getRandBoxWith(fake_x, boxestoInc=bbox), classify=train_edit_mode)

                    d_loss_fake = self.compute_real_fake_loss(out_src_fake, self.adv_loss_type,datasrc = 'fake')

                    if self.adv_classifier:
                        d_loss_cls = d_loss_cls + (F.binary_cross_entropy_with_logits(
                            out_cls_fake.view(cbsz,-1), boxlabel, size_average=False) / cbsz)

                    # Backward + Optimize
                    if not self.use_seperate_classifier:
                        d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls
                    else:
                        d_loss = d_loss_real + d_loss_fake

                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    # Compute gradient penalty
                    if not self.d_use_spectralnorm:
                        d_loss_gp = self.compute_gradient_penalty(real_x, fake_x, bbox=bbox)
                        # Backward + Optimize
                        d_loss = self.lambda_gp * d_loss_gp
                        self.reset_grad()
                        d_loss.backward()
                        self.d_optimizer.step()
                        loss['D/l_gp']   = loss['D/l_gp']   * avgexp + (1-avgexp ) * d_loss_gp.data[0]   if 'D/_gp'   in loss else d_loss_gp.data[0]

                    loss['W_dist'] = loss['W_dist'] * avgexp + (1-avgexp ) * (torch.mean(out_src_sel) - torch.mean(out_src_fake)).data[0] if 'W_dist' in loss else (torch.mean(out_src_sel) - torch.mean(out_src_fake)).data[0]
                    # Logging
                    loss['D/l_real'] = loss['D/l_real'] * avgexp + (1-avgexp ) * d_loss_real.data[0] if 'D/l_real' in loss else d_loss_real.data[0]
                    loss['D/l_fake'] = loss['D/l_fake'] * avgexp + (1-avgexp ) * d_loss_fake.data[0] if 'D/l_fake' in loss else d_loss_fake.data[0]
                    #loss['D/loss_cls']  = loss['D/loss_cls']  * avgexp + (1-avgexp ) * d_loss_cls.data[0]  if 'D/loss_cls'  in loss else d_loss_cls.data[0]
                    #del grad, interpolated, out, out_cls, grad_l2norm, d_loss_gp, alpha
                    #gc.collect()

                # ================== Train G ================== #
                #if ((((i+1) % self.d_train_repeat) == 0) and (e-start)>0) or self.only_reconst_loss:
                if ((((i+1) % self.d_train_repeat) == 0)) or self.only_reconst_loss:
                    for p in self.G.parameters():
                        p.requires_grad = bool(not train_edit_mode or self.g_fine_tune)
                    if self.E is not None:
                        for p in self.E.parameters():
                            p.requires_grad = True#

                    for p in self.D.parameters():
                        p.requires_grad = False#

                    # Original-to-target and Target-to-original domain
                    #if ('G/loss_rec' in loss) and (loss['G/loss_rec'] < 0.22):
                    #    import ipdb; ipdb.set_trace()

                    # When training in auto encoder mode, use the real labels.
                    # When trying to do editing, us fake label.
                    fake_x = self.forward_boxreconst(real_x, boxImg, mask, fake_boxlabel_inp if train_edit_mode else boxlabel, bbox)

                    # Use reconstruction loss to teach adding objects.
                    # All editing is done on the same box. But here use only the discriminator to decide which object to add.
                    if train_edit_mode and self.lambda_rec:
                        rec_x = self.forward_boxreconst(fake_x.detach(), None, mask, fake_boxlabel_inp_rec , bbox)
                        if self.adv_rec:
                            out_src_rec, out_cls_rec,_ = self.D(rec_x, self.getRandBoxWith(fake_x, bbox), True)

                    # Compute losses
                    if not self.only_reconst_loss:
                        out_src, out_cls, out_feat_fake = self.D(fake_x, self.getRandBoxWith(fake_x, bbox), classify= train_edit_mode)
                        g_loss_fake = self.compute_real_fake_loss(out_src, self.adv_loss_type,loss_for = 'generator')
                    else:
                        g_loss_fake = 0.

                    # Compute the reconstruction error within bbox
                    g_loss_cls = 0.
                    if self.lambda_rec:
                        # In this mode the input image is the ground truth target.
                        if train_edit_mode:
                            g_loss_rec = F.smooth_l1_loss(rec_x,real_x) + 1.*self.vggLoss(rec_x, real_x)
                            if self.adv_rec:
                                g_loss_rec = g_loss_rec - torch.mean(out_src_rec) + 2.*F.binary_cross_entropy_with_logits(out_cls_rec.view(cbsz,-1), boxlabel, size_average=False)/ cbsz
                        else:
                            g_loss_rec = F.smooth_l1_loss(fake_x,real_x)

                    if train_edit_mode:
                        # In this mode there is no ground truth. Just rely on the discriminator's real/fake signal
                        g_loss_cls = F.binary_cross_entropy_with_logits(
                            out_cls.view(cbsz,-1), fake_boxlabel, size_average=False)/ cbsz
                        loss['G/l_cls'] = loss['G/l_cls'] * avgexp + (1-avgexp ) * (g_loss_cls.data[0]) if 'G/l_vgg' in loss else (g_loss_cls.data[0])

                    # Backward + Optimize
                    g_loss = 2*g_loss_fake + self.lambda_rec * g_loss_rec + g_loss_cls
                    if self.lambda_vggloss:
                        if train_edit_mode:
                            # Compute perception loss only on images which don't have person
                            g_loss_vgg = self.vggLoss(fake_x[z_idx,::], real_x[z_idx,::])
                        else:
                            g_loss_vgg = self.vggLoss(fake_x, real_x)

                        g_loss = g_loss + self.lambda_vggloss * g_loss_vgg
                        loss['G/l_vgg'] = loss['G/l_vgg'] * avgexp + (1-avgexp ) * (self.lambda_vggloss * g_loss_vgg.data[0]) if 'G/l_vgg' in loss else (self.lambda_vggloss*g_loss_vgg.data[0])

                    if self.lambda_feat_match:
                        _, _, out_feat_real  = self.D(real_x, self.getRandBoxWith(real_x, bbox))
                        # Match mean of features, not individual features!!
                        if train_edit_mode:
                            # Compare fake features only to real images not containing the object
                            g_loss_feat = F.mse_loss(out_feat_fake.mean(dim=0), out_feat_real[z_idx,::].detach().mean(dim=0))
                        else:
                            g_loss_feat = F.mse_loss(out_feat_fake.mean(dim=0), out_feat_real.detach().mean(dim=0))

                        g_loss = g_loss + self.lambda_feat_match * g_loss_feat
                        loss['G/l_feat'] = loss['G/l_feat'] * avgexp + (1-avgexp) * (self.lambda_feat_match* g_loss_feat.data[0]) if 'G/l_feat' in loss else (self.lambda_feat_match * g_loss_feat.data[0])

                    # Add L1 loss to keep the difference image from going beyond boundaries
                    self.reset_grad()
                    g_loss.backward()
                    if (not train_edit_mode or self.g_fine_tune):
                        self.g_optimizer.step()
                    if self.E is not None:
                        self.e_optimizer.step()

                    # Logging
                    if not self.only_reconst_loss:
                        loss['G/l_fake'] = loss['G/l_fake'] * avgexp + (1-avgexp ) * g_loss_fake.data[0] if 'G/l_fake' in loss else g_loss_fake.data[0]
                    if self.lambda_rec:
                        loss['G/l_rec']  = loss['G/l_rec']  * avgexp + (1-avgexp ) * self.lambda_rec*g_loss_rec.data[0]  if 'G/l_rec'  in loss else self.lambda_rec*g_loss_rec.data[0]
                    #loss['G/loss_cls']  = loss['G/loss_cls']  * avgexp + (1-avgexp ) * g_loss_cls.data[0]  if 'G/loss_cls'  in loss else g_loss_cls.data[0]
                    for p in self.D.parameters():
                        p.requires_grad = True#
                    if self.E is not None:
                        for p in self.E.parameters():
                            p.requires_grad = False#
                    for p in self.G.parameters():
                        p.requires_grad = False#

                    if (self.D_cls is not None) and self.use_seperate_classifier:
                        for p in self.D_cls.parameters():
                            p.requires_grad = True#


                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.3f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

            # Translate fixed images for debugging
            if (e+1) % self.sample_step == 0:
                fake_image_list = [fixed_x, fixed_mask.expand(*fixed_x.size())]
                fake_image_list.append(self.forward_boxreconst(fixed_x, fixed_box, fixed_mask, fixed_boxlabel, fixed_bbox))
                fake_images = torch.cat(fake_image_list, dim=3)
                save_image(self.denorm(fake_images.data),
                    os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
                print('Translated images and saved into {}..!'.format(self.sample_path))

            # Save model checkpoints
            if (e+1) % self.model_save_step == 0:
                checkpointData = {
                    'iter': e*iters_per_epoch+i,
                    'arch': self.arch,
                    'generator_state_dict':self.G.state_dict(),
                    'discriminator_state_dict': self.D.state_dict(),
                    'gen_optimizer' : self.g_optimizer.state_dict(),
                    'discriminator_optimizer' : self.d_optimizer.state_dict(),
                }
                if self.E is not None:
                    checkpointData['encoder_state_dict'] = self.E.state_dict(),
                    checkpointData['encoder_optimizer']  = self.e_optimizer.state_dict(),

                save_checkpoint(checkpointData, fappend = self.arch['fappend']+'_{}_{}.pth.tar'.format(e+1,i+1),
                    outdir = self.model_save_path)

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(10.))
                d_lr -= (self.d_lr / float(10.))
                e_lr -= (self.e_lr / float(10.))
                g_lr = max(1e-6, g_lr)
                d_lr = max(1e-5, d_lr)
                e_lr = max(1e-5, e_lr)
                self.update_lr(g_lr, d_lr, e_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_multi(self):
        """Train StarGAN with multiple datasets.
        In the code below, 1 is related to CelebA and 2 is releated to RaFD.
        """
        fixed_x2 = []
        for i, (images, labels) in enumerate(self.rafd_loader):
            fixed_x2.append(images)
            if i == 2:
                break

        fixed_x2 = torch.cat(fixed_x2, dim=0)
        fixed_x2 = self.to_var(fixed_x2, volatile=True)

        # Fixed imagse and labels for debugging
        fixed_x = []
        real_c = []

        for i, (images, labels) in enumerate(self.celebA_loader):
            fixed_x.append(images)
            real_c.append(labels)
            if i == 2:
                break

        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)
        real_c = torch.cat(real_c, dim=0)
        fixed_c1_list = self.make_celeb_labels(real_c)

        fixed_c2_list = []
        for i in range(self.c2_dim):
            fixed_c = self.one_hot(torch.ones(fixed_x.size(0)) * i, self.c2_dim)
            fixed_c2_list.append(self.to_var(fixed_c, volatile=True))

        fixed_zero1 = self.to_var(torch.zeros(fixed_x.size(0), self.c2_dim))     # zero vector when training with CelebA
        fixed_mask1 = self.to_var(self.one_hot(torch.zeros(fixed_x.size(0)), 2)) # mask vector: [1, 0]
        fixed_zero2 = self.to_var(torch.zeros(fixed_x.size(0), self.c_dim))      # zero vector when training with RaFD
        fixed_mask2 = self.to_var(self.one_hot(torch.ones(fixed_x.size(0)), 2))  # mask vector: [0, 1]

        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr

        # data iterator
        data_iter1 = iter(self.celebA_loader)
        data_iter2 = iter(self.rafd_loader)

        # Start with trained model
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[-2]) + 1
        else:
            start = 0

        # # Start training
        start_time = time.time()
        for i in range(start, self.num_iters):
            # Fetch mini-batch images and labels
            try:
                real_x1, real_label1 = next(data_iter1)
            except:
                data_iter1 = iter(self.celebA_loader)
                real_x1, real_label1 = next(data_iter1)

            try:
                real_x2, real_label2 = next(data_iter2)
            except:
                data_iter2 = iter(self.rafd_loader)
                real_x2, real_label2 = next(data_iter2)

            # Generate fake labels randomly (target domain labels)
            rand_idx = torch.randperm(real_label1.size(0))
            fake_label1 = real_label1[rand_idx]
            rand_idx = torch.randperm(real_label2.size(0))
            fake_label2 = real_label2[rand_idx]

            real_c1 = real_label1.clone()
            fake_c1 = fake_label1.clone()
            zero1 = torch.zeros(real_x1.size(0), self.c2_dim)
            mask1 = self.one_hot(torch.zeros(real_x1.size(0)), 2)

            real_c2 = self.one_hot(real_label2, self.c2_dim)
            fake_c2 = self.one_hot(fake_label2, self.c2_dim)
            zero2 = torch.zeros(real_x2.size(0), self.c_dim)
            mask2 = self.one_hot(torch.ones(real_x2.size(0)), 2)

            # Convert tensor to variable
            real_x1 = self.to_var(real_x1)
            real_c1 = self.to_var(real_c1)
            fake_c1 = self.to_var(fake_c1)
            mask1 = self.to_var(mask1)
            zero1 = self.to_var(zero1)

            real_x2 = self.to_var(real_x2)
            real_c2 = self.to_var(real_c2)
            fake_c2 = self.to_var(fake_c2)
            mask2 = self.to_var(mask2)
            zero2 = self.to_var(zero2)

            real_label1 = self.to_var(real_label1)
            fake_label1 = self.to_var(fake_label1)
            real_label2 = self.to_var(real_label2)
            fake_label2 = self.to_var(fake_label2)

            # ================== Train D ================== #

            # Real images (CelebA)
            out_real, out_cls = self.D(real_x1)
            out_cls1 = out_cls[:, :self.c_dim]      # celebA part
            d_loss_real = - torch.mean(out_real)
            d_loss_cls = F.binary_cross_entropy_with_logits(out_cls1, real_label1, size_average=False) / real_x1.size(0)

            # Real images (RaFD)
            out_real, out_cls = self.D(real_x2)
            out_cls2 = out_cls[:, self.c_dim:]      # rafd part
            d_loss_real += - torch.mean(out_real)
            d_loss_cls += F.cross_entropy(out_cls2, real_label2)

            # Compute classification accuracy of the discriminator
            if (i+1) % self.log_step == 0:
                accuracies = self.compute_accuracy(out_cls1, real_label1, 'CelebA')
                log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                print(log)
                accuracies = self.compute_accuracy(out_cls2, real_label2, 'RaFD')
                log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                print(log)

            # Fake images (CelebA)
            fake_c = torch.cat([fake_c1, zero1, mask1], dim=1)
            fake_x1 = self.G(real_x1, fake_c)
            fake_x1 = Variable(fake_x1.data)
            out_fake, _ = self.D(fake_x1)
            d_loss_fake = torch.mean(out_fake)

            # Fake images (RaFD)
            fake_c = torch.cat([zero2, fake_c2, mask2], dim=1)
            fake_x2 = self.G(real_x2, fake_c)
            out_fake, _ = self.D(fake_x2)
            d_loss_fake += torch.mean(out_fake)

            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Compute gradient penalty
            if (i+1) % 2 == 0:
                real_x = real_x1
                fake_x = fake_x1
            else:
                real_x = real_x2
                fake_x = fake_x2

            alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
            interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
            out, out_cls = self.D(interpolated)

            if (i+1) % 2 == 0:
                out_cls = out_cls[:, :self.c_dim]  # CelebA
            else:
                out_cls = out_cls[:, self.c_dim:]  # RaFD

            grad = torch.autograd.grad(outputs=out,
                                       inputs=interpolated,
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1)**2)

            # Backward + Optimize
            d_loss = self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging
            loss = {}
            loss['D/loss_real'] = d_loss_real.data[0]
            loss['D/loss_fake'] = d_loss_fake.data[0]
            loss['D/loss_cls'] = d_loss_cls.data[0]
            loss['D/loss_gp'] = d_loss_gp.data[0]

            # ================== Train G ================== #
            if (i+1) % self.d_train_repeat == 0:
                # Original-to-target and target-to-original domain (CelebA)
                fake_c = torch.cat([fake_c1, zero1, mask1], dim=1)
                real_c = torch.cat([real_c1, zero1, mask1], dim=1)
                fake_x1 = self.G(real_x1, fake_c)
                rec_x1 = self.G(fake_x1, real_c)

                # Compute losses
                out, out_cls = self.D(fake_x1)
                out_cls1 = out_cls[:, :self.c_dim]
                g_loss_fake = - torch.mean(out)
                g_loss_rec = torch.mean(torch.abs(real_x1 - rec_x1))
                g_loss_cls = F.binary_cross_entropy_with_logits(out_cls1, fake_label1, size_average=False) / fake_x1.size(0)

                # Original-to-target and target-to-original domain (RaFD)
                fake_c = torch.cat([zero2, fake_c2, mask2], dim=1)
                real_c = torch.cat([zero2, real_c2, mask2], dim=1)
                fake_x2 = self.G(real_x2, fake_c)
                rec_x2 = self.G(fake_x2, real_c)

                # Compute losses
                out, out_cls = self.D(fake_x2)
                out_cls2 = out_cls[:, self.c_dim:]
                g_loss_fake += - torch.mean(out)
                g_loss_rec += torch.mean(torch.abs(real_x2 - rec_x2))
                g_loss_cls += F.cross_entropy(out_cls2, fake_label2)

                # Backward + Optimize
                g_loss = g_loss_fake + self.lambda_cls * g_loss_cls + self.lambda_rec * g_loss_rec
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging
                loss['G/loss_fake'] = g_loss_fake.data[0]
                loss['G/loss_cls'] = g_loss_cls.data[0]
                loss['G/loss_rec'] = g_loss_rec.data[0]

            # Print out log info
            if (i+1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))

                log = "Elapsed [{}], Iter [{}/{}]".format(
                    elapsed, i+1, self.num_iters)

                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate the images (debugging)
            if (i+1) % self.sample_step == 0:
                fake_image_list = [fixed_x]

                # Changing hair color, gender, and age
                for j in range(self.c_dim):
                    fake_c = torch.cat([fixed_c1_list[j], fixed_zero1, fixed_mask1], dim=1)
                    fake_image_list.append(self.G(fixed_x, fake_c))
                # Chaning emotional expressions
                for j in range(self.c2_dim):
                    fake_c = torch.cat([fixed_zero2, fixed_c2_list[j], fixed_mask2], dim=1)
                    fake_image_list.append(self.G(fixed_x, fake_c))
                fake = torch.cat(fake_image_list, dim=3)

                # Save the translated images
                save_image(self.denorm(fake.data),
                    os.path.join(self.sample_path, '{}_fake.png'.format(i+1)), nrow=1, padding=0)

            # Save model checkpoints
            if (i+1) % self.model_save_step == 0:
                torch.save(self.G.state_dict(),
                    os.path.join(self.model_save_path, '{}_G.pth'.format(i+1)))
                torch.save(self.D.state_dict(),
                    os.path.join(self.model_save_path, '{}_D.pth'.format(i+1)))

            # Decay learning rate
            decay_step = 1000
            if (i+1) > (self.num_iters - self.num_iters_decay) and (i+1) % decay_step==0:
                g_lr -= (self.g_lr / float(self.num_iters_decay) * decay_step)
                d_lr -= (self.d_lr / float(self.num_iters_decay) * decay_step)
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()

        for i, (real_x, real_c) in enumerate(self.data_loader):
            real_x = self.to_var(real_x, volatile=True)
            target_c_list = self.make_celeb_labels(real_c)

            # Start translations
            fake_image_list = [real_x]
            for target_c in target_c_list:
                fake_image_list.append(self.G(real_x, target_c))
            fake_images = torch.cat(fake_image_list, dim=3)
            save_image(self.denorm(fake_images.data),
                os.path.join(self.sample_path, '{}_fake.png'.format(i+1)),nrow=1, padding=0)
            print('Translated test images and saved into {}..!'.format(self.result_path))
