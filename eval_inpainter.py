import argparse
import numpy as np
import time
import torch
import json
import torch.nn as nn
import torch.nn.functional as FN
import cv2
import random
from tqdm import tqdm

from solver import Solver
from removalmodels.models import Generator, Discriminator
from removalmodels.models import GeneratorDiff, GeneratorDiffWithInp, GeneratorDiffAndMask, GeneratorDiffAndMask_V2, VGGLoss
from os.path import basename, exists, join, splitext
from os import makedirs
from torch.autograd import Variable
from utils.data_loader_stargan import get_dataset
from torch.backends import cudnn
from utils.utils import show
from skimage.measure import compare_ssim, compare_psnr

class ParamObject(object):

    def __init__(self, adict):
        """Convert a dictionary to a class

        @param :adict Dictionary
        """

        self.__dict__.update(adict)

        for k, v in adict.items():
            if isinstance(v, dict):
                self.__dict__[k] = ParamObject(v)

    def __getitem__(self,key):
        return self.__dict__[key]

    def values(self):
        return self.__dict__.values()

    def itemsAsDict(self):
        return dict(self.__dict__.items())

def VOCap(rec,prec):

    nc = rec.shape[1]
    mrec=np.concatenate([np.zeros((1,rec.shape[1])),rec,np.ones((1,rec.shape[1]))],axis=0)
    mprec=np.concatenate([np.zeros((1,rec.shape[1])),prec,np.zeros((1,rec.shape[1]))],axis=0)
    for i in reversed(np.arange(mprec.shape[0]-1)):
        mprec[i,:]=np.maximum(mprec[i,:],mprec[i+1,:])

    #-------------------------------------------------------
    # Now do the step wise integration
    # Original matlab code is
    #-------------------------------------------------------
    # i=find(mrec(2:end)~=mrec(1:end-1))+1;
    # ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    # Here we use boolean indexing of numpy instead of find
    steps = (mrec[1:,:] != mrec[:-1,:])
    ap = np.zeros(nc)
    for i in xrange(nc):
        ap[i]=sum((mrec[1:,:][steps[:,i], i] - mrec[:-1,:][steps[:,i], i])*mprec[1:,][steps[:,i],i])
    return ap


def computeAP(allSc, allLb):
    si = (-allSc).argsort(axis=0)
    cid = np.arange(20)
    tp = allLb[si[:,cid],cid] > 0.
    fp = allLb[si[:,cid],cid] == 0.
    tp = tp.cumsum(axis=0).astype(np.float32)
    fp = fp.cumsum(axis=0).astype(np.float32)
    rec = tp/(allLb>0.).sum(axis=0).astype(np.float32)
    prec = tp/ (tp+ fp)
    ap = VOCap(rec,prec)

    return ap

def get_sk_image(img):
    img = img[:,[0,0,0], ::] if img.shape[1] == 1 else img
    img = np.clip(img.data.cpu().numpy().transpose(0, 2, 3, 1),-1,1)
    img  = 255*((img[0,::] + 1) / 2)
    return img


def gen_samples(params):
    # For fast training
    #cudnn.benchmark = True
    gpu_id = 0
    use_cuda = params['cuda']
    b_sz  = params['batch_size']

    solvers = []
    configs = []
    for i, mfile in enumerate(params['model']):
        model = torch.load(mfile)
        configs.append(model['arch'])
        configs[-1]['pretrained_model'] = mfile
        configs[-1]['load_encoder'] = 1
        configs[-1]['load_discriminator'] = 0
        configs[-1]['image_size'] = params['image_size']
        if i==0:
            configs[i]['onlypretrained_discr'] = params['evaluating_discr']
        else:
            configs[i]['onlypretrained_discr'] = None

        if params['withExtMask'] and params['mask_size']!= 32:
            configs[-1]['lowres_mask'] = 0
            configs[-1]['load_encoder'] = 0

        solvers.append(Solver(None, None, ParamObject(configs[-1]), mode='test' if i > 0 else 'eval', pretrainedcv=model))
        solvers[-1].G.eval()
        if configs[-1]['train_boxreconst'] >0:
            solvers[-1].E.eval()

    solvers[0].D.eval()
    solvers[0].D_cls.eval()

    dataset = get_dataset('', '', params['image_size'], params['image_size'], params['dataset'], params['split'],
                          select_attrs=configs[0]['selected_attrs'], datafile=params['datafile'], bboxLoader=1,
                          bbox_size = params['box_size'], randomrotate = params['randomrotate'],
                          randomscale=params['randomscale'], max_object_size=params['max_object_size'],
                          use_gt_mask = 0, n_boxes = params['n_boxes'])#configs[0]['use_gtmask_inp'])#, imagenet_norm=(configs[0]['use_imagenet_pretrained'] is not None))

    #gt_mask_data = get_dataset('','', params['mask_size'], params['mask_size'], params['dataset'], params['split'],
    #                         select_attrs=configs[0]['selected_attrs'], bboxLoader=0, loadMasks = True)
    #data_iter = DataLoader(targ_split, batch_size=b_sz, shuffle=True, num_workers=8)
    targ_split =  dataset #train if params['split'] == 'train' else valid if params['split'] == 'val' else test
    data_iter = np.random.permutation(len(targ_split) if params['nImages'] == -1 else params['nImages'])

    if params['withExtMask'] or params['computeSegAccuracy']:
        gt_mask_data = get_dataset('','', params['mask_size'], params['mask_size'],
                                   params['dataset'] if params['extMask_source']=='gt' else params['extMask_source'],
                                   params['split'], select_attrs=configs[0]['selected_attrs'], bboxLoader=0, loadMasks = True)
        commonIds = set(gt_mask_data.valid_ids).intersection(set(dataset.valid_ids))
        commonIndexes = [i for i in xrange(len(dataset.valid_ids)) if dataset.valid_ids[i] in commonIds]
        data_iter = commonIndexes if params['nImages'] == -1 else commonIndexes[:params['nImages']]

    print('-----------------------------------------')
    print('%s'%(' | '.join(targ_split.selected_attrs)))
    print('-----------------------------------------')

    flatten = lambda l: [item for sublist in l for item in sublist]
    selected_attrs = configs[0]['selected_attrs']

    if params['showreconst'] and len(params['names'])>0:
        params['names'] = flatten([[nm,nm+'-R'] for nm in params['names']])

    #discriminator.load_state_dict(cv['discriminator_state_dict'])
    c_idx = 0
    np.set_printoptions(precision=2)
    padimg = np.zeros((params['image_size'],5,3),dtype=np.uint8)
    padimg[:,:,:] = 128
    vggLoss = VGGLoss(network='squeeze')
    cimg_cnt = 0

    mask_bin_size = 0.1
    n_bins = int(1.0/mask_bin_size)
    vLTotal = np.zeros((n_bins,))
    pSNRTotal = np.zeros((n_bins,))
    ssimTotal = np.zeros((n_bins,))
    total_count = np.zeros((n_bins,)) + 1e-8

    perImageRes = {'images':{}, 'overall':{}}
    if params['dilateMask']:
        dilateWeight = torch.ones((1,1,params['dilateMask'],params['dilateMask']))
        dilateWeight = Variable(dilateWeight,requires_grad=False).cuda()
    else:
        dilateWeight = None


    for i in tqdm(xrange(len(data_iter))):
    #for i in tqdm(xrange(2)):
        idx = data_iter[i]
        x, real_label, boxImg, boxlabel, mask, bbox, curCls  = targ_split[idx]
        cocoid = targ_split.getcocoid(idx)
        nnz_cls = real_label.nonzero()
        z_cls = (1-real_label).nonzero()

        z_cls = z_cls[:,0]
        x = x[None,::]; boxImg = boxImg[None,::]; mask = mask[None,::]; boxlabel = boxlabel[None,::]; real_label = real_label[None,::]

        x, boxImg, mask, boxlabel = solvers[0].to_var(x, volatile=True), solvers[0].to_var(boxImg, volatile=True), solvers[0].to_var(mask, volatile=True), solvers[0].to_var(boxlabel, volatile=True)
        fake_x, mask_out = solvers[0].forward_generator(x, imagelabel = None, mask_threshold=params['mask_threshold'], onlyMasks=False, mask=mask, withGTMask=True, dilate = dilateWeight)
        vL = vggLoss(fake_x, x).data[0]
        # Change the image range to 0, 255
        fake_x_sk = get_sk_image(fake_x)
        x_sk = get_sk_image(x)
        pSNR = compare_psnr(fake_x_sk,x_sk,data_range = 255.)
        ssim = compare_ssim(fake_x_sk,x_sk,data_range = 255., multichannel=True)
        msz = mask.data.cpu().numpy().mean()
        if msz > 0.:
            msz_bin = int((msz-1e-8)/mask_bin_size)

            perImageRes['images'][cocoid] = {'overall':{}}
            perImageRes['images'][cocoid]['overall']['perceptual'] = float(vL)
            perImageRes['images'][cocoid]['overall']['pSNR'] = float(pSNR)
            perImageRes['images'][cocoid]['overall']['ssim'] = float(ssim)
            perImageRes['images'][cocoid]['overall']['mask_size'] = float(msz)
            vLTotal[msz_bin] += vL
            pSNRTotal[msz_bin] += pSNR
            ssimTotal[msz_bin] += ssim
            total_count[msz_bin] += 1

    print '------------------------------------------------------------'
    print '                Metrics have been computed                  '
    print '------------------------------------------------------------'
    print('Percp: || %s |'%(' | '.join(['  %.3f' % sc for sc in [vLTotal.sum()/total_count.sum()] + list(vLTotal/total_count)])))
    print('pSNR : || %s |'%(' | '.join(['  %.3f' % sc for sc in [pSNRTotal.sum()/total_count.sum()] + list(pSNRTotal/total_count)])))
    print('ssim : || %s |'%(' | '.join(['  %.3f' % sc for sc in [ssimTotal.sum()/total_count.sum()] + list(ssimTotal/total_count)])))
    if params['dump_perimage_res']:
        json.dump(perImageRes, open(join(params['dump_perimage_res'], params['split']+'_'+ basename(params['model'][0]).split('.')[0]),'w'))


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--showdiff', type=int, default=0)
  parser.add_argument('--showperceptionloss', type=int, default=0)
  parser.add_argument('--showdeform', type=int, default=0)
  parser.add_argument('--showmask', type=int, default=0)
  #parser.add_argument('--showclassifier', type=int, default=0)
  parser.add_argument('--showreconst', type=int, default=0)
  parser.add_argument('--mask_threshold', type=float, default=0.3)
  parser.add_argument('-d', '--dataset', dest='dataset',  type=str, default='coco', help='dataset: celeb')
  parser.add_argument('-m', '--model', type=str, default=[], nargs='+', help='checkpoint to resume training from')
  parser.add_argument('-n', '--names', type=str, default=[], nargs='+', help='checkpoint to resume training from')
  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=1, help='max batch size')
  parser.add_argument('--sample_dump_dir', type=str, default='gen_samples', help='print every x iters')
  parser.add_argument('--swap_attr', type=str, default='rand', help='which attribute to swap')
  parser.add_argument('--split', type=str, default='val', help='which attribute to swap')
  parser.add_argument('--nImages', type=int, default=-1)
  parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
  parser.add_argument('--max_object_size', type=float, default=0.3)

  parser.add_argument('--dump_perimage_res', type=str, default=None, help='perImageResults')
  parser.add_argument('--evaluating_discr', type=str, default=None)
  parser.add_argument('--eval_only_discr', type=int, default=0)
  parser.add_argument('--withExtMask', type=int, default=0)
  parser.add_argument('--computeSegAccuracy', type=int, default=0)
  parser.add_argument('--dump_cls_results', type=int, default=0)
  parser.add_argument('--extMask_source', type=str, default='gt')
  parser.add_argument('--dilateMask', type=int, default=0)

  # Deformations applied to mnist images;
  parser.add_argument('--randomrotate', type=int, default=90)
  parser.add_argument('--randomscale', type=float, nargs='+', default=[0.5,0.5])
  parser.add_argument('--image_size', type=int, default=128)
  parser.add_argument('--mask_size', type=int, default=32)
  parser.add_argument('--scaleDisp', type=int, default=0)
  parser.add_argument('--box_size', type=int, default=64)
  parser.add_argument('--computeAP', type=int, default=1)
  parser.add_argument('--datafile', type=str, default='datasetBoxAnn_80pcMaxObj.json')
  parser.add_argument('--n_boxes', type=int, default=4)

  parser.add_argument('--compute_deform_stats', type=int, default=0)

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  params['cuda'] = not args.no_cuda
  print json.dumps(params, indent = 2)
  gen_samples(params)

