import os
import argparse
from solver import Solver
from utils.data_loader_stargan import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # Data loader
    data_loader = None
    rafd_loader = None
    bboxLoader = ((config.train_boxreconst==1) or (config.train_boxreconst==2)) if config.use_bbox_loader == None else config.use_bbox_loader

    if config.dataset in ['CelebA', 'Both', 'coco', 'mnist', 'celebbox', 'pascal', 'places2', 'flickrlogo', 'belgalogo', 'ade20k']:
        data_loader = get_loader(config.celebA_image_path, config.metadata_path, config.celebA_crop_size, config.image_size,
                                 config.batch_size, config.dataset, config.mode, select_attrs=config.selected_attrs,
                                 datafile=config.datafile, bboxLoader=bboxLoader,
                                 bbox_size = config.box_size, randomrotate = config.randomrotate, randomscale=config.randomscale,
                                 balance_classes = config.balance_classes, onlyrandBoxes=config.use_random_boxes, max_object_size=config.max_object_size,
                                 imagenet_norm = False, use_gt_mask = config.use_gtmask_inp, n_boxes = config.n_boxes) # imagenet_norm is set to false now as I experimentally verified that adjusting the mean and variance inside the module works just as well.

        config.selected_attrs = data_loader.dataset.selected_attrs
    if config.dataset in ['RaFD', 'Both']:
        rafd_loader = get_loader(config.rafd_image_path, None, config.rafd_crop_size,
                                 config.image_size, config.batch_size, 'RaFD', config.mode)

    if config.use_maskprior_gan:
        mask_loader = get_loader(config.celebA_image_path, config.metadata_path, config.celebA_crop_size,
                                 config.mask_size, config.batch_size, config.maskdataset,
                                 config.mode, select_attrs=config.selected_attrs, datafile=config.datafile,
                                 loadMasks = True, balance_classes=config.balance_classes, n_masks=config.n_masks_perclass)
    else:
        mask_loader = None
    #c = 0
    #for i, (real_x, real_label) in enumerate(celebA_loader):
    #    c = c+1
    #print c
    solver = Solver(data_loader, rafd_loader, config, mask_loader = mask_loader)
    if config.mode == 'train':
        if config.train_boxreconst==3:
            solver.train_fulleditor()
        elif config.train_boxreconst>0:
            solver.train_reconst_nw()
        else:
            if config.dataset in ['CelebA', 'RaFD', 'coco', 'pascal', 'places2', 'flickrlogo', 'belgalogo', 'ade20k']:
                solver.train()
            elif config.dataset in ['Both']:
                solver.train_multi()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--differential_generator', type=int, default=0)
    parser.add_argument('--onlymaskgen', type=int, default=0)
    parser.add_argument('--maskpaint', type=int, default=0)
    parser.add_argument('--use_maskprior_gan', type=int, default=1)
    parser.add_argument('--use_gtmask_inp', type=int, default=0)
    parser.add_argument('--boxprior', type=int, default=0)
    parser.add_argument('--n_masks_perclass', type=int, default=-1)
    parser.add_argument('--n_boxes', type=int, default=1)
    parser.add_argument('--maskprior_matchclass', type=int, default=1)
    parser.add_argument('--maskdataset', type=str, default='coco')
    parser.add_argument('--alternate_mask_train', type=int, default=2)
    parser.add_argument('--train_g_every', type=int, default=2)
    parser.add_argument('--use_unmasked_input', type=int, default=0)
    parser.add_argument('--use_bbox_loader', type=int, default=1)
    parser.add_argument('--discrim_masked_image', type=int, default=0)

    # Model hyper-parameters
    parser.add_argument('--c_dim', type=int, default=20)
    parser.add_argument('--c2_dim', type=int, default=8)
    parser.add_argument('--celebA_crop_size', type=int, default=178)
    parser.add_argument('--rafd_crop_size', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--g_smooth_layers', type=int, default=0)
    parser.add_argument('--g_binary_mask', type=int, default=0)
    parser.add_argument('--gen_fullimage', type=int, default=0)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--e_repeat_num', type=int, default=5)
    parser.add_argument('--d_init_stride', type=int, default=2)
    parser.add_argument('--d_max_filters', type=int, default=512)
    parser.add_argument('--d_kernel_size', type=int, default=3)
    parser.add_argument('--d_max_filters_cls', type=int, default=512)
    parser.add_argument('--d_global_pool', type=str, default='mean')
    parser.add_argument('--d_use_spectralnorm', type=int, default=0)
    parser.add_argument('--d_patch_size', type=int, default=16)

    parser.add_argument('--g_downsamp_layers', type=int, default=3)
    parser.add_argument('--g_upsample_type', type=str, default='bilinear')
    parser.add_argument('--g_nupsampFilters', type=int, default=2)
    parser.add_argument('--g_pad_type', type=str, default='zeros')
    parser.add_argument('--g_dil_start', type=int, default=1)
    parser.add_argument('--g_fixed_size', type=int, default=0)
    parser.add_argument('--e_norm_type', type=str, default='instance')
    parser.add_argument('--e_ksize', type=int, default=4)

    parser.add_argument('--e_addnoise', type=int, default=0)
    parser.add_argument('--e_masked_image', type=int, default=0)
    parser.add_argument('--e_use_residual', type=int, default=0)
    parser.add_argument('--e_bias', action='store_true', default=False)
    parser.add_argument('--lowres_mask', type=int, default=0)
    parser.add_argument('--mask_size', type=int, default=32)
    parser.add_argument('--mask_additional_cond', type=str, default='image')
    parser.add_argument('--perclass_mask', type=int, default=1)
    parser.add_argument('--mask_noinplabel', type=int, default=0)
    parser.add_argument('--mask_normalize_byclass', type=int, default=0)
    parser.add_argument('--m_upsample_type', type=str, default='nearest')

    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--e_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--use_topk_patch', type=int, default=0)
    parser.add_argument('--lambda_rec', type=float, default=0.)
    parser.add_argument('--lambda_feat_match', type=float, default=0)
    parser.add_argument('--lambda_vggloss', type=float, default=0)
    parser.add_argument('--use_style_loss', type=float, default=0)
    parser.add_argument('--lambda_smoothloss', type=float, default=0)
    parser.add_argument('--lambda_maskL1loss', type=float, default=0)
    parser.add_argument('--lambda_tvloss', type=float, default=0.)
    parser.add_argument('--grad_weighted_l1loss', type=float, default=0)
    parser.add_argument('--lambda_maskfake_loss', type=float, default=1.)
    parser.add_argument('--vggloss_nw', type=str, default='vgg')
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--onlytargetclsloss', type=int, default=1)
    parser.add_argument('--useclsweights', type=int, default=0)
    parser.add_argument('--clsweight_scale_decay', type=int, default=0)
    parser.add_argument('--clsweight_scale_init', type=float, default=1.)

    parser.add_argument('--max_object_size', type=float, default=0.4)

    parser.add_argument('--adv_rec', type=float, default=0)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--d_train_repeat', type=int, default=5)
    parser.add_argument('--selected_attrs', type=str, nargs='+', default=['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train', 'bottle', 'couch', "dining table", "potted plant", 'chair', 'tv'])
    #parser.add_argument('--selected_attrs', type=str, nargs='+', default=[])
    parser.add_argument('--balance_classes', type=int, default=1)

    # Seperate discriminator parameters
    parser.add_argument('--use_seperate_classifier', type=int, default=1)
    parser.add_argument('--use_gap_classifier', type=int, default=1)
    parser.add_argument('--use_tv_inp', type=int, default=0)
    parser.add_argument('--use_imagenet_pretrained', type=str, default='vgg19')
    parser.add_argument('--use_imagenet_pretrained_mask', type=str, default=None)
    parser.add_argument('--use_imnetmask_v2', type=int, default=1)
    parser.add_argument('--cond_inp_pnet', type=int, default=0)
    parser.add_argument('--cond_parallel_track', type=int, default=0)
    parser.add_argument('--use_bnorm', type=int, default=1)
    parser.add_argument('--use_bnorm_mask', type=int, default=2)
    parser.add_argument('--adv_classifier', type=int, default=1)
    parser.add_argument('--adv_loss_type', type=str, default='lsgan')
    parser.add_argument('--m_adv_loss_type', type=str, default='wgan')

    # Training settings
    parser.add_argument('--dataset', type=str, default='coco', choices=['CelebA', 'RaFD', 'Both', 'coco', 'mnist', 'celebbox', 'pascal', 'places2',
        'flickrlogo', 'belgalogo', 'ade20k'])
    parser.add_argument('--datafile', type=str, default='datasetBoxAnn_80pcMaxObj.json')
    parser.add_argument('--only_remove_train', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_epochs_decay', type=int, default=10)
    parser.add_argument('--decay_every', type=int, default=5)
    parser.add_argument('--num_iters', type=int, default=200000)
    parser.add_argument('--num_iters_decay', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Deformations applied to mnist images;
    parser.add_argument('--randomrotate', type=int, default=90)
    parser.add_argument('--randomscale', type=float, nargs='+', default=[0.4,0.4])

    parser.add_argument('--train_boxreconst', type=int, default=0)
    parser.add_argument('--g_fine_tune', type=int, default=1)
    parser.add_argument('--no_inpainter', type=int, default=0)
    parser.add_argument('--fixed_m', type=int, default=0)
    parser.add_argument('--only_gt_mask', type=int, default=0)
    parser.add_argument('--dilateMask', type=int, default=0)
    parser.add_argument('--train_g_wo_m', type=int, default=0)
    parser.add_argument('--train_only_g', type=int, default=0)
    parser.add_argument('--fixed_classifier', type=int, default=0)
    parser.add_argument('--train_only_d', type=int, default=0)
    parser.add_argument('--train_robust_d', type=int, default=0)
    parser.add_argument('--compositional_loss', type=float, default=0.)
    parser.add_argument('--full_image_encoder', type=int, default=0)
    parser.add_argument('--only_reconst_loss', type=int, default=0)
    parser.add_argument('--use_box_label', type=int, default=0)

    parser.add_argument('--box_size', type=int, default=64)
    parser.add_argument('--boxfeat_dim', type=int, default=0)

    parser.add_argument('--onlypretrained_discr', type=str, default=None)
    parser.add_argument('--onlypretrained_encoder', type=str, default=None)
    parser.add_argument('--onlypretrained_generator', type=str, default=None)
    parser.add_argument('--load_generator', type=int, default=1)
    parser.add_argument('--load_encoder', type=int, default=1)
    parser.add_argument('--load_discriminator', type=int, default=1)
    parser.add_argument('--use_random_boxes', type=int, default=0)
    parser.add_argument('--use_past_masks', type=int, default=0)
    parser.add_argument('--only_random_boxes_discr', type=int, default=0)
    parser.add_argument('--d_local_supervision', type=int, default=1)

    # Test settings
    parser.add_argument('--test_model', type=str, default='20_1000')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--celebA_image_path', type=str, default='./data/celebA/images')
    parser.add_argument('--rafd_image_path', type=str, default='./data/RaFD/train')
    parser.add_argument('--metadata_path', type=str, default='./data/celebA/list_attr_celeba.txt')
    parser.add_argument('--log_path', type=str, default='./test/logs')
    parser.add_argument('--model_save_path', type=str, default='./stargancv/models')
    parser.add_argument('--sample_path', type=str, default='./stargancv/samples')
    parser.add_argument('--result_path', type=str, default='./stargancv/results')
    parser.add_argument('--fappend', dest='fappend', type=str, default='baseline', help='append this string to checkpoint filenames')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=2)


    config = parser.parse_args()
    config.sample_path = os.path.join(config.sample_path,config.fappend)
    config.result_path = os.path.join(config.result_path,config.fappend)
    config.model_save_path = os.path.join(config.model_save_path,config.fappend)

    print(config)

    main(config)
