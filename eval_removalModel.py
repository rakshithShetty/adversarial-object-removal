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

def get_sk_image(img):
    img = img[:,[0,0,0], ::] if img.shape[1] == 1 else img
    img = np.clip(img.data.cpu().numpy().transpose(0, 2, 3, 1),-1,1)
    img  = 255*((img[0,::] + 1) / 2)
    return img
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
    cid = np.arange(allLb.shape[1])
    tp = allLb[si[:,cid],cid] > 0.
    fp = allLb[si[:,cid],cid] == 0.
    tp = tp.cumsum(axis=0).astype(np.float32)
    fp = fp.cumsum(axis=0).astype(np.float32)
    rec = (tp+1e-8)/((allLb>0.)+1e-8).sum(axis=0).astype(np.float32)
    prec = (tp+1e-8)/ (tp+ fp+1e-8)
    ap = VOCap(rec,prec)

    return ap


def gen_samples(params):
    # For fast training
    #cudnn.benchmark = True
    gpu_id = 0
    use_cuda = params['cuda']
    b_sz  = params['batch_size']

    if params['use_same_g']:
        if len(params['use_same_g']) == 1:
           gCV = torch.load(params['use_same_g'][0])
    solvers = []
    configs = []
    for i, mfile in enumerate(params['model']):
        model = torch.load(mfile)
        configs.append(model['arch'])
        configs[-1]['pretrained_model'] = mfile
        configs[-1]['load_encoder'] = 1
        configs[-1]['load_discriminator'] = 0 if params['evaluating_discr'] is not None else 1
        if i==0:
            configs[i]['onlypretrained_discr'] = params['evaluating_discr']
        else:
            configs[i]['onlypretrained_discr'] = None

        if params['withExtMask'] and params['mask_size']!= 32:
            configs[-1]['lowres_mask'] = 0
            configs[-1]['load_encoder'] = 0
        else:
            params['mask_size'] = 32

        solvers.append(Solver(None, None, ParamObject(configs[-1]), mode='test' if i > 0 else 'eval', pretrainedcv=model))
        solvers[-1].G.eval()
        if configs[-1]['train_boxreconst'] >0:
            solvers[-1].E.eval()
        if params['use_same_g']:
            solvers[-1].no_inpainter = 0
            solvers[-1].load_pretrained_generator(gCV)
            print 'loaded generator again'

    solvers[0].D.eval()
    solvers[0].D_cls.eval()

    dataset = get_dataset('', '', params['image_size'], params['image_size'], params['dataset'], params['split'],
                          select_attrs=configs[0]['selected_attrs'], datafile=params['datafile'], bboxLoader=1,
                          bbox_size = params['box_size'], randomrotate = params['randomrotate'],
                          randomscale=params['randomscale'], max_object_size=params['max_object_size'],
                          use_gt_mask = configs[0]['use_gtmask_inp'], onlyrandBoxes= params['extmask_type'] == 'randbox',
                          square_resize=configs[0].get('square_resize',0) if params['square_resize_override'] < 0 else params['square_resize_override'], filter_by_mincooccur= params['filter_by_mincooccur'],
                          only_indiv_occur = params['only_indiv_occur'])

    #gt_mask_data = get_dataset('','', params['mask_size'], params['mask_size'], params['dataset'], params['split'],
    #                         select_attrs=configs[0]['selected_attrs'], bboxLoader=0, loadMasks = True)
    #data_iter = DataLoader(targ_split, batch_size=b_sz, shuffle=True, num_workers=8)
    targ_split =  dataset #train if params['split'] == 'train' else valid if params['split'] == 'val' else test
    data_iter = np.random.permutation(len(targ_split))

    if params['computeSegAccuracy']:
        gt_mask_data = get_dataset('','', params['mask_size'], params['mask_size'],
                                   params['dataset'],
                                   params['split'], select_attrs=configs[0]['selected_attrs'], bboxLoader=0, loadMasks = True)
        commonIds = set(gt_mask_data.valid_ids).intersection(set(dataset.valid_ids))
        commonIndexes = [i for i in xrange(len(dataset.valid_ids)) if dataset.valid_ids[i] in commonIds]
        data_iter = commonIndexes

    if params['withExtMask'] and (params['extmask_type'] == 'mask'):
        ext_mask_data = get_dataset('','', params['mask_size'], params['mask_size'],
                                   params['dataset'] if params['extMask_source']=='gt' else params['extMask_source'],
                                   params['split'], select_attrs=configs[0]['selected_attrs'], bboxLoader=0, loadMasks = True)
        curr_valid_ids = [dataset.valid_ids[i] for i in data_iter]
        commonIds = set(ext_mask_data.valid_ids).intersection(set(curr_valid_ids))
        commonIndexes = [i for i in xrange(len(dataset.valid_ids)) if dataset.valid_ids[i] in commonIds]
        data_iter = commonIndexes

    if params['nImages'] > -1:
        data_iter = data_iter[:params['nImages']]


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

    perclass_removeSucc = np.zeros((len(selected_attrs)))
    perclass_confusion = np.zeros((len(selected_attrs), len(selected_attrs)))
    perclass_classScoreDrop = np.zeros((len(selected_attrs), len(selected_attrs)))
    perclass_cooccurence = np.zeros((len(selected_attrs), len(selected_attrs))) + 1e-6
    perclass_vgg = np.zeros((len(selected_attrs)))
    perclass_ssim = np.zeros((len(selected_attrs)))
    perclass_psnr = np.zeros((len(selected_attrs)))
    perclass_tp = np.zeros((len(selected_attrs)))
    perclass_fp = np.zeros((len(selected_attrs)))
    perclass_fn = np.zeros((len(selected_attrs)))
    perclass_acc = np.zeros((len(selected_attrs)))
    perclass_counts = np.zeros((len(selected_attrs))) + 1e-6
    perclass_int = np.zeros((len(selected_attrs)))
    perclass_union = np.zeros((len(selected_attrs)))
    perclass_gtsize = np.zeros((len(selected_attrs)))
    perclass_predsize = np.zeros((len(selected_attrs)))
    perclass_segacc = np.zeros((len(selected_attrs)))
    perclass_msz = np.zeros((len(selected_attrs)))
    #perclass_th = Variable(torch.FloatTensor(np.array([0., 0.5380775, -0.49303985, -0.48941165, 2.8394265, -0.37880898, 1.0709367, 1.6613332, -1.5602279, 1.2631614, 2.4104881, -0.29175103, -0.6607682, -0.2128999, -1.286599, -2.24577, -0.4130093, -1.0535073, 0.038890466, -0.6808476]))).cuda()

    perclass_th = Variable(torch.FloatTensor(np.zeros((len(selected_attrs))))).cuda()

    perImageRes = {'images':{}, 'overall':{}}
    total_count = 0.
    if params['computeAP']:
        allScores = []
        allGT = []
        allEditedSc = []
    if params['dilateMask']:
        dilateWeight = torch.ones((1,1,params['dilateMask'],params['dilateMask']))
        dilateWeight = Variable(dilateWeight,requires_grad=False).cuda()
    else:
        dilateWeight = None

    all_masks = []
    all_imgidAndCls = []

    for i in tqdm(xrange(len(data_iter))):
    #for i in tqdm(xrange(2)):
        idx = data_iter[i]
        x, real_label, boxImg, boxlabel, mask, bbox, curCls  = targ_split[idx]
        cocoid = targ_split.getcocoid(idx)
        nnz_cls = real_label.nonzero()
        z_cls = (1-real_label).nonzero()

        z_cls = z_cls[:,0] if len(z_cls.size()) > 1 else z_cls
        x = x[None,::]; boxImg = boxImg[None,::]; mask = mask[None,::]; boxlabel = boxlabel[None,::]; real_label = real_label[None,::]

        x, boxImg, mask, boxlabel = solvers[0].to_var(x, volatile=True), solvers[0].to_var(boxImg, volatile=True), solvers[0].to_var(mask, volatile=True), solvers[0].to_var(boxlabel, volatile=True)
        real_label = solvers[0].to_var(real_label, volatile=True)
        _, out_cls_real = solvers[0].classify(x)
        out_cls_real = out_cls_real[0]# Remove the singleton dimension
        pred_real_label = (out_cls_real > perclass_th)
        total_count += 1
        #;import ipdb; ipdb.set_trace()

        if params['computeAP']:
            allScores.append(out_cls_real[None,:])
            allGT.append(real_label)
            removeScores = out_cls_real.clone()

        perclass_acc[(pred_real_label.float() == real_label)[0,:].data.cpu().numpy().astype(np.bool)] += 1.
        if len(z_cls):
            perclass_fp[z_cls.numpy()] += pred_real_label.data.cpu()[z_cls]
        if len(nnz_cls):
            nnz_cls = nnz_cls[:,0]
            perclass_tp[nnz_cls.numpy()] += pred_real_label.data.cpu()[nnz_cls]
            perclass_fn[nnz_cls.numpy()] += 1-pred_real_label.data.cpu()[nnz_cls]

            perImageRes['images'][cocoid] = {'perclass': {}}
            if params['dump_cls_results']:
                perImageRes['images'][cocoid]['real_label'] =  nnz_cls.tolist()
                perImageRes['images'][cocoid]['real_scores'] = out_cls_real.data.cpu().tolist()
            if not params['eval_only_discr']:
                for cid in nnz_cls:
                    if configs[0]['use_gtmask_inp']:
                        mask = solvers[0].to_var(targ_split.getGTMaskInp(idx, configs[0]['selected_attrs'][cid])[None,::], volatile=True)
                    if params['withExtMask']:
                        if params['extmask_type'] == 'mask':
                            mask = solvers[0].to_var(ext_mask_data.getbyIdAndclass(cocoid,configs[0]['selected_attrs'][cid])[None,::], volatile=True)
                        elif params['extmask_type'] == 'box':
                            mask = solvers[0].to_var(dataset.getGTMaskInp(idx,configs[0]['selected_attrs'][cid], mask_type=2)[None,::],volatile=True)
                        elif params['extmask_type'] == 'randbox':
                            # Nothing to do here, mask is already set to random boxes
                            None
                    if params['computeSegAccuracy']:
                        gtMask = gt_mask_data.getbyIdAndclass(cocoid,configs[0]['selected_attrs'][cid]).cuda()
                    mask_target = torch.zeros_like(real_label)
                    fake_label = real_label.clone()
                    fake_label[0,cid] = 0.
                    mask_target[0,cid] = 1
                    fake_x, mask_out = solvers[0].forward_generator(x, imagelabel = mask_target, mask_threshold=params['mask_threshold'], onlyMasks=False, mask=mask, withGTMask=params['withExtMask'], dilate = dilateWeight)
                    _, out_cls_fake = solvers[0].classify(fake_x)
                    out_cls_fake = out_cls_fake[0]# Remove the singleton dimension
                    mask_out = mask_out.data[0,::]

                    if params['dump_mask']:
                        all_masks.append(mask_out.cpu().numpy())
                        all_imgidAndCls.append((cocoid,selected_attrs[cid]))


                    perImageRes['images'][cocoid]['perclass'][selected_attrs[cid]] = {}
                    if params['computeSegAccuracy']:
                        union = torch.clamp((gtMask + mask_out),max=1.0).sum()
                        intersection = (gtMask * mask_out).sum()
                        img_iou = (intersection/(union+1e-6))
                        img_acc = (gtMask == mask_out).float().mean()
                        img_recall = ((intersection/(gtMask.sum()+1e-6)))
                        img_precision = (intersection/(mask_out.sum()+1e-6))
                        perImageRes['images'][cocoid]['perclass'][selected_attrs[cid]].update({'iou': img_iou, 'rec':img_recall, 'prec': img_precision, 'acc': img_acc})
                        perImageRes['images'][cocoid]['perclass'][selected_attrs[cid]]['gtSize'] =  gtMask.mean()
                        perImageRes['images'][cocoid]['perclass'][selected_attrs[cid]]['predSize'] =  mask_out.mean()

                        # Compute metrics now
                        perclass_counts[cid] += 1
                        perclass_int[cid] += intersection
                        perclass_union[cid] += union
                        perclass_gtsize[cid] += gtMask.sum()
                        perclass_predsize[cid] += mask_out.sum()
                        perclass_segacc[cid] += img_acc
                    if params['dump_cls_results']:
                        perImageRes['images'][cocoid]['perclass'][selected_attrs[cid]]['remove_scores'] = out_cls_fake.data.cpu().tolist()
                    perImageRes['images'][cocoid]['perclass'][selected_attrs[cid]]['diff'] = out_cls_real.data[cid] - out_cls_fake.data[cid]

                    remove_succ = float((out_cls_fake.data[cid] < perclass_th[cid]))# and (out_cls_real[cid]>0.))
                    perclass_removeSucc[cid] += remove_succ
                    vL = vggLoss(fake_x, x).data[0]
                    perclass_vgg[cid] += 100.*vL

                    fake_x_sk = get_sk_image(fake_x)
                    x_sk = get_sk_image(x)
                    pSNR = compare_psnr(fake_x_sk,x_sk,data_range = 255.)
                    ssim = compare_ssim(fake_x_sk,x_sk,data_range = 255., multichannel=True)
                    msz = mask_out.mean()
                    if msz > 0.:
                        perclass_ssim[cid] += ssim
                        perclass_psnr[cid] += pSNR

                    if params['computeAP']:
                        removeScores[cid] = out_cls_fake[cid]

                    #---------------------------------------------------------------
                    # These are classes not trying to be removed;
                    # correctly detect on real image and not detected on fake image
                    # This are collateral damage. Count these
                    #---------------------------------------------------------------
                    false_remove = fake_label.byte()*(out_cls_fake<perclass_th)*(out_cls_real>perclass_th)
                    perclass_cooccurence[cid, nnz_cls.numpy()] += 1.
                    perclass_confusion[cid,false_remove.data.cpu().numpy().astype(np.bool)[0,:]] += 1

                    perImageRes['images'][cocoid]['perclass'][selected_attrs[cid]].update({'remove_succ':remove_succ, 'false_remove': float(false_remove.data.cpu().float().numpy().sum()), 'perceptual': 100.*vL})
                if params['computeAP']:
                    allEditedSc.append(removeScores[None,:])

                perImageRes['images'][cocoid]['overall'] = {}
                perImageRes['images'][cocoid]['overall']['remove_succ'] = np.mean([perImageRes['images'][cocoid]['perclass'][cls]['remove_succ'] for cls in perImageRes['images'][cocoid]['perclass']])
                perImageRes['images'][cocoid]['overall']['false_remove'] = np.mean([perImageRes['images'][cocoid]['perclass'][cls]['false_remove'] for cls in perImageRes['images'][cocoid]['perclass']])
                perImageRes['images'][cocoid]['overall']['perceptual'] = np.mean([perImageRes['images'][cocoid]['perclass'][cls]['perceptual'] for cls in perImageRes['images'][cocoid]['perclass']])
                perImageRes['images'][cocoid]['overall']['diff'] = np.mean([perImageRes['images'][cocoid]['perclass'][cls]['diff'] for cls in perImageRes['images'][cocoid]['perclass']])
                if params['computeSegAccuracy']:
                    perImageRes['images'][cocoid]['overall']['iou'] = np.mean([perImageRes['images'][cocoid]['perclass'][cls]['iou'] for cls in perImageRes['images'][cocoid]['perclass']])
                    perImageRes['images'][cocoid]['overall']['acc'] = np.mean([perImageRes['images'][cocoid]['perclass'][cls]['acc'] for cls in perImageRes['images'][cocoid]['perclass']])
                    perImageRes['images'][cocoid]['overall']['prec'] = np.mean([perImageRes['images'][cocoid]['perclass'][cls]['prec'] for cls in perImageRes['images'][cocoid]['perclass']])
                    perImageRes['images'][cocoid]['overall']['rec'] = np.mean([perImageRes['images'][cocoid]['perclass'][cls]['rec'] for cls in perImageRes['images'][cocoid]['perclass']])

        elif params['dump_cls_results']:
            perImageRes['images'][cocoid] = {'perclass': {}}
            perImageRes['images'][cocoid]['real_label'] =  nnz_cls.tolist()
            perImageRes['images'][cocoid]['real_scores'] = out_cls_real.data.cpu().tolist()

    if params['dump_mask']:
        np.savez('allMasks.npz', masks=np.concatenate(all_masks).astype(np.uint8), idAndClass=np.stack(all_imgidAndCls))

    if params['computeAP']:
        allScores = torch.cat(allScores,dim=0).data.cpu().numpy()
        allGT= torch.cat(allGT,dim=0).data.cpu().numpy()
        apR = computeAP(allScores, allGT)
        if not params['eval_only_discr']:
            allEditedSc= torch.cat(allEditedSc,dim=0).data.cpu().numpy()
            apEdited = computeAP(allEditedSc, allGT)
    #for i in xrange(len(selected_attrs)):
    #    pr,rec,th = precision_recall_curve(allGTArr[:,i],allPredArr[:,i]);
    #    f1s = 2*(pr*rec)/(pr+rec); mf1idx = np.argmax(f1s);
    #    #print 'Max f1 = %.2f, th =%.2f'%(f1s[mf1idx], th[mf1idx]);
    #    allMf1s.append(f1s[mf1idx])
    #    allTh.append(th[mf1idx])
    recall = perclass_tp/(perclass_tp+perclass_fn+1e-6)
    precision = perclass_tp/(perclass_tp+perclass_fp+1e-6)
    f1_score = 2.0* (recall*precision)/(recall+precision+1e-6)

    present_classes =  (perclass_tp+perclass_fn)>0.
    perclass_gt_counts = (perclass_tp+perclass_fn)
    apROverall = (perclass_gt_counts*apR).sum() / (perclass_gt_counts.sum())
    apR = apR[present_classes]
    recall = recall[present_classes]
    f1_score = f1_score[present_classes]
    precision = precision[present_classes]
    perclass_acc = perclass_acc[present_classes]
    present_attrs = [att for i, att in enumerate(targ_split.selected_attrs) if present_classes[i]]


    rec_overall = perclass_tp.sum()/ (perclass_tp.sum() + perclass_fn.sum() + 1e-6)
    prec_overall = perclass_tp.sum()/ (perclass_tp.sum() + perclass_fp.sum() + 1e-6)
    f1_score_overall = 2.0* (rec_overall*prec_overall)/(rec_overall+prec_overall+1e-6)
    print '------------------------------------------------------------'
    print '                Metrics have been computed                  '
    print '------------------------------------------------------------'
    print('Score: || %s |'%(' | '.join(['%6s'%att[:6] for att in ['Overall', 'OverCls']+present_attrs])))
    print('Acc  : || %s |'%(' | '.join(['  %.2f' % sc for sc in [(perclass_acc/total_count).mean()]+[(perclass_acc/total_count).mean()]+list(perclass_acc/total_count)])))
    print('F1-sc: || %s |'%(' | '.join(['  %.2f' % sc for sc in [f1_score_overall]+[f1_score.mean()]+list(f1_score)])))
    print('recal: || %s |'%(' | '.join(['  %.2f' % sc for sc in [rec_overall]+[recall.mean()]+list(recall)])))
    print('prec : || %s |'%(' | '.join(['  %.2f' % sc for sc in [prec_overall]+[precision.mean()]+list(precision)])))
    if params['computeAP']:
        print('AP   : || %s |'%(' | '.join(['  %.2f' % sc for sc in [apROverall]+[apR.mean()]+list(apR)])))
    print('Count: || %s |'%(' | '.join(['  %4.0f' % sc for sc in [perclass_gt_counts.mean()]+[perclass_gt_counts.mean()]+list(perclass_gt_counts[present_classes])])))
    if not params['eval_only_discr']:
        print('R-suc: || %s |'%(' | '.join(['  %.2f' % sc for sc in [(perclass_removeSucc.sum()/perclass_cooccurence.diagonal().sum())]+[(perclass_removeSucc/perclass_cooccurence.diagonal()).mean()]+list(perclass_removeSucc/perclass_cooccurence.diagonal())])))
        print('R-fal: || %s |'%(' | '.join(['  %.2f' % sc for sc in [(perclass_confusion.sum()/(perclass_cooccurence.sum() - perclass_cooccurence.diagonal().sum()))]+[(perclass_confusion.sum(axis=1)/(perclass_cooccurence.sum(axis=1) - perclass_cooccurence.diagonal())).mean()]+list((perclass_confusion/perclass_cooccurence).sum(axis=1)/(perclass_cooccurence.shape[0]-1))])))
        print('Percp: || %s |'%(' | '.join(['  %.2f' % sc for sc in [(perclass_vgg.sum()/perclass_cooccurence.diagonal().sum())]+[(perclass_vgg/perclass_cooccurence.diagonal()).mean()]+list(perclass_vgg/perclass_cooccurence.diagonal())])))
        print('pSNR : || %s |'%(' | '.join([' %.2f' % sc for sc in [(perclass_psnr.sum()/perclass_cooccurence.diagonal().sum())]+[(perclass_psnr/perclass_cooccurence.diagonal()).mean()]+list(perclass_psnr/perclass_cooccurence.diagonal())])))
        print('ssim : || %s |'%(' | '.join([' %.3f' % sc for sc in [(perclass_ssim.sum()/perclass_cooccurence.diagonal().sum())]+[(perclass_ssim/perclass_cooccurence.diagonal()).mean()]+list(perclass_ssim/perclass_cooccurence.diagonal())])))
        if params['computeAP']:
            print('R-AP : || %s |'%(' | '.join(['  %.2f' % sc for sc in [apEdited.mean()]+[apEdited.mean()]+list(apEdited)])))


        if params['computeSegAccuracy']:
            print('mIou : || %s |'%(' | '.join(['  %.2f' % sc for sc in [(perclass_int.sum()/(perclass_union+1e-6).sum())]+[(perclass_int/(perclass_union+1e-6)).mean()]+list(perclass_int/(perclass_union+1e-6))])))
            print('mRec : || %s |'%(' | '.join(['  %.2f' % sc for sc in [(perclass_int.sum()/(perclass_gtsize+1e-6).sum())]+[(perclass_int/(perclass_gtsize+1e-6)).mean()]+list(perclass_int/(perclass_gtsize+1e-6))])))
            print('mPrc : || %s |'%(' | '.join(['  %.2f' % sc for sc in [(perclass_int.sum()/(perclass_predsize.sum()))]+[(perclass_int/(perclass_predsize+1e-6)).mean()]+list(perclass_int/(perclass_predsize+1e-6))])))
            print('mSzR : || %s |'%(' | '.join(['  %.2f' % sc for sc in [(perclass_predsize.sum()/(perclass_gtsize.sum()))]+[(perclass_predsize/(perclass_gtsize+1e-6)).mean()]+list(perclass_predsize/(perclass_gtsize+1e-6))])))
            print('Acc  : || %s |'%(' | '.join(['  %.2f' % sc for sc in [(perclass_segacc.sum()/(perclass_counts.sum()))]+[(perclass_segacc/(perclass_counts+1e-6)).mean()]+list(perclass_segacc/(perclass_counts+1e-6))])))
            print('mSz  : || %s |'%(' | '.join(['  %.1f' % sc for sc in [(100.*(perclass_predsize.sum()/(params['mask_size']*params['mask_size']*perclass_counts).sum()))]+[(100.*(perclass_predsize/(params['mask_size']*params['mask_size']*perclass_counts+1e-6))).mean()]+list((100.*perclass_predsize)/(params['mask_size']*params['mask_size']*perclass_counts+1e-6))])))

        perImageRes['overall'] = {'iou': 0., 'rec': 0., 'prec':0., 'acc':0.}
        perImageRes['overall']['remove_succ'] =(perclass_removeSucc/perclass_cooccurence.diagonal()).mean()
        perImageRes['overall']['false_remove'] =(perclass_confusion/perclass_cooccurence).mean()
        perImageRes['overall']['perceptual'] =(perclass_vgg/perclass_cooccurence.diagonal()).mean()
        if params['computeSegAccuracy']:
            perImageRes['overall']['iou'] =(perclass_int/(perclass_union+1e-6)).mean()
            perImageRes['overall']['acc'] =(perclass_segacc/(perclass_counts+1e-6)).mean()
            perImageRes['overall']['prec'] =(perclass_int/(perclass_predsize+1e-6)).mean()
            perImageRes['overall']['psize'] =(perclass_predsize).mean()
            perImageRes['overall']['psize_rel'] =(perclass_predsize/(perclass_gtsize+1e-6)).mean()
            perImageRes['overall']['rec'] =(perclass_int/(perclass_gtsize+1e-6)).mean()
        if params['computeAP']:
            perImageRes['overall']['ap-orig'] = list(apR)
            perImageRes['overall']['ap-edit'] = list(apEdited)

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
  parser.add_argument('--filter_by_mincooccur', type=float, default=-1.)
  parser.add_argument('--only_indiv_occur', type=float, default=0)
  parser.add_argument('--square_resize_override',type=int, default=-1)

  parser.add_argument('--dump_perimage_res', type=str, default=None, help='perImageResults')
  parser.add_argument('--evaluating_discr', type=str, default=None)
  parser.add_argument('--eval_only_discr', type=int, default=0)
  parser.add_argument('--withExtMask', type=int, default=0)
  parser.add_argument('--extmask_type', type=str, default='mask')
  parser.add_argument('--computeSegAccuracy', type=int, default=0)
  parser.add_argument('--dump_cls_results', type=int, default=0)
  parser.add_argument('--extMask_source', type=str, default='gt')
  parser.add_argument('--dilateMask', type=int, default=0)
  parser.add_argument('--dump_mask', type=int, default=0)
  parser.add_argument('--use_same_g', type=str, default=[], nargs='+', help='Evaluation scores to visualize')

  # Deformations applied to mnist images;
  parser.add_argument('--randomrotate', type=int, default=0)
  parser.add_argument('--randomscale', type=float, nargs='+', default=[0.5,0.5])
  parser.add_argument('--image_size', type=int, default=128)
  parser.add_argument('--mask_size', type=int, default=32)
  parser.add_argument('--scaleDisp', type=int, default=0)
  parser.add_argument('--box_size', type=int, default=64)
  parser.add_argument('--computeAP', type=int, default=1)
  parser.add_argument('--datafile', type=str, default='datasetBoxAnn_80pcMaxObj.json')

  parser.add_argument('--compute_deform_stats', type=int, default=0)

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  params['cuda'] = not args.no_cuda
  print json.dumps(params, indent = 2)
  gen_samples(params)
