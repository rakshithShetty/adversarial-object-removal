import argparse
import numpy as np
import time
import torch
import json
import torch.nn as nn
import cv2
import random

from solver import Solver
from removalmodels.models import Generator, Discriminator
from removalmodels.models import GeneratorDiff, GeneratorDiffWithInp, GeneratorDiffAndMask, GeneratorDiffAndMask_V2, VGGLoss
from os.path import basename, exists, join, splitext
from os import makedirs
from torch.autograd import Variable
from utils.data_loader_stargan import get_dataset
from torch.backends import cudnn
import operator
from collections import OrderedDict

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


def make_image_with_text(img_size, text):
    fVFrm = 255*np.ones(img_size,dtype=np.uint8)
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(fVFrm, text,(4,img_size[0]-10), font, 0.8,(0,0,0), 1,cv2.LINE_AA)
    return fVFrm

def make_coco_labels(real_c):
    """Generate domain labels for CelebA for debugging/testing.

    if dataset == 'CelebA':
        return single and multiple attribute changes
    elif dataset == 'Both':
        return single attribute changes
    """
    y = np.eye(real_c.size(1))

    fixed_c_list = []

    # single object addition and removal
    for i in range(2*real_c.size(1)):
        fixed_c = real_c.clone()
        for c in fixed_c:
            if i%2:
                c[i//2] = 0.
            else:
                c[i//2] = 1.
        fixed_c_list.append(Variable(fixed_c, volatile=True).cuda())

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

def make_celeb_labels(real_c, c_dim=5, dataset='CelebA'):
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
    for i in range(c_dim):
        fixed_c = real_c.clone()
        for c in fixed_c:
            if i < 3:
                c[:3] = y[i]
            else:
                c[i] = 0 if c[i] == 1 else 1   # opposite value
        fixed_c_list.append(Variable(fixed_c, volatile=True).cuda())

    # multi-attribute transfer (H+G, H+A, G+A, H+G+A)
    if dataset == 'CelebA':
        for i in range(4):
            fixed_c = real_c.clone()
            for c in fixed_c:
                if i in [0, 1, 3]:   # Hair color to brown
                    c[:3] = y[2]
                if i in [0, 2, 3]:   # Gender
                    c[3] = 0 if c[3] == 1 else 1
                if i in [1, 2, 3]:   # Aged
                    c[4] = 0 if c[4] == 1 else 1
            fixed_c_list.append(Variable(fixed_c, volatile=True).cuda())
    return fixed_c_list



def make_image(img_list, padimg=None):
    edit_images = []

    for img in img_list:
        img = img[:,[0,0,0], ::] if img.shape[1] == 1 else img
        img = np.clip(img.data.cpu().numpy().transpose(0, 2, 3, 1),-1,1)
        img  = 255*((img[0,::] + 1) / 2)
        edit_images.append(img)
        if padimg is not None:
            edit_images.append(padimg)
        #img_out = 255 * ((x_hat[i] + 1) / 2)
        #img_out_flip = 255 * ((x_hat_flip[i] + 1) / 2)
        #img_diff = np.clip(5*np.abs(img_out - img_out_flip), 0, 255)
        #img = Image.fromarray(stacked.astype(np.uint8))
        #img = Image.fromarray(stacked.astype(np.uint8))
    #stacked = stacked.transpose(2,1,0)
    stacked = np.hstack((edit_images))

    stacked = cv2.cvtColor(stacked.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return stacked

def simple_make_image(img):
    img = img[:,[0,0,0], ::] if img.shape[1] == 1 else img
    img = np.clip(img.data.cpu().numpy().transpose(0, 2, 3, 1),-1,1)
    img = 255*((img[0,::] + 1) / 2)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return img

def saveIndividImages(image_list, mask_image_list, nameList,sample_dir, fp, cls):
    #sample_dir = join(params['sample_dump_dir'], basename(params['model'][0]).split('.')[0])
    fdir = join(sample_dir, splitext(basename(fp[0]))[0]+'_'+cls)
    if not exists(fdir):
        makedirs(fdir)

    for i, img in enumerate(image_list):
        fname = join(fdir, nameList[i]+'.png')
        img = simple_make_image(img)
        cv2.imwrite(fname, img)
        print 'Saving into file: ' + fname

    if mask_image_list is not None:
        for i, img in enumerate(mask_image_list):
            # Skip the first one. It is just empty image
            if i > 0:
                fname = join(fdir, 'mask_'+nameList[i]+'.png')
                img = simple_make_image(img)
                cv2.imwrite(fname, img)
                print 'Saving into file: ' + fname

def draw_arrows(img, pt1 , pt2):
    imgSz = img.shape[0]
    pt1 = ((pt1.data.cpu().numpy()+1.)/2.) * imgSz
    pt2 = ((pt2.data.cpu().numpy()[0,::]+1.)/2.) * imgSz
    for i in xrange(0,pt1.shape[1],2):
        for j in xrange(0,pt1.shape[2],2):
            if np.abs(pt1[0,i,j]-pt2[0,i,j]) > 2. or  np.abs(pt1[1,i,j]-pt2[1,i,j]) > 2. :
                img = cv2.arrowedLine(img.astype(np.uint8), tuple(pt1[:,i,j]), tuple(pt2[:,i,j]), color=(0,0,255), line_type=cv2.LINE_AA, thickness=1, tipLength = 0.4)
    return img

def make_image_with_deform(img_list, deformList, padimg=None):
    edit_images = []

    for i, img in enumerate(img_list):
        img = img[:,[0,0,0], ::] if img.shape[1] == 1 else img
        img = np.clip(img.data.cpu().numpy().transpose(0, 2, 3, 1),-1,1)
        img = 255*((img[0,::] + 1) / 2)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cur_deform=[]
        if len(deformList[i])>0 and len(deformList[i][0])>0:
            for d in deformList[i]:
                cur_deform.append(draw_arrows(img, d[1], d[0]))
        else:
            cur_deform=[img*0, img*0, img*0]

        edit_images.append(np.vstack(cur_deform))
        if padimg is not None:
            edit_images.append(padimg)
    stacked = np.hstack((edit_images))

    return stacked

def compute_deform_statistics(pt1 , pt2):
    imgSz = 128
    pt1 = ((pt1.data.cpu().numpy()+1.)/2.) * imgSz
    pt2 = ((pt2.data.cpu().numpy()[0,::]+1.)/2.) * imgSz
    lengths = np.linalg.norm(pt1-pt2,axis=0).flatten()
    mean = lengths.mean()
    maxl = lengths.max()

    return lengths, mean, maxl

def gen_samples(params):
    # For fast training
    #cudnn.benchmark = True
    gpu_id = 0
    use_cuda = params['cuda']
    b_sz  = params['batch_size']

    g_conv_dim = 64
    d_conv_dim = 64
    c_dim= 5
    c2_dim = 8
    g_repeat_num= 6
    d_repeat_num= 6
    select_attrs=[]

    if params['use_same_g']:
        if len(params['use_same_g']) == 1:
           gCV = torch.load(params['use_same_g'][0])
    solvers = []
    configs = []
    for i,mfile in enumerate(params['model']):
        model = torch.load(mfile)
        configs.append(model['arch'])
        configs[-1]['pretrained_model'] = mfile
        configs[-1]['load_encoder'] = 1
        configs[-1]['load_discriminator'] = 0
        configs[-1]['image_size'] = params['image_size']
        if 'g_downsamp_layers' not in configs[-1]:
            configs[-1]['g_downsamp_layers'] = 2
        if 'g_dil_start' not in configs[-1]:
            configs[-1]['g_dil_start'] = 0
            configs[-1]['e_norm_type'] = 'drop'
            configs[-1]['e_ksize'] = 4
        if len(params['withExtMask']) and params['mask_size']!= 32:
            if params['withExtMask'][i]:
                configs[-1]['lowres_mask'] = 0
                configs[-1]['load_encoder'] = 0

        solvers.append(Solver(None, None, ParamObject(configs[-1]), mode='test', pretrainedcv=model))
        solvers[-1].G.eval()
        #solvers[-1].D.eval()
        if configs[-1]['train_boxreconst'] >0 and solvers[-1].E is not None:
            solvers[-1].E.eval()
        if params['use_same_g']:
            solvers[-1].load_pretrained_generator(gCV)

    if len(params['dilateMask']):
        assert(len(params['model']) == len(params['dilateMask']))
        dilateWeightAll = []
        for di in xrange(len(params['dilateMask'])):
            if params['dilateMask'][di] > 0:
                dilateWeight = torch.ones((1,1,params['dilateMask'][di],params['dilateMask'][di]))
                dilateWeight = Variable(dilateWeight,requires_grad=False).cuda()
            else:
                dilateWeight = None
            dilateWeightAll.append(dilateWeight)
    else:
        dilateWeightAll = [None for i in xrange(len(params['model']))]

    dataset = get_dataset('', '', params['image_size'], params['image_size'], params['dataset'], params['split'],
                          select_attrs=configs[0]['selected_attrs'], datafile=params['datafile'], bboxLoader=1,
                          bbox_size = params['box_size'], randomrotate = params['randomrotate'],
                          randomscale = params['randomscale'], max_object_size=params['max_object_size'],
                          use_gt_mask = configs[0]['use_gtmask_inp'], n_boxes = params['n_boxes']
                          , onlyrandBoxes= (params['extmask_type'] == 'randbox'))
    #data_iter = DataLoader(targ_split, batch_size=b_sz, shuffle=True, num_workers=8)
    targ_split =  dataset #train if params['split'] == 'train' else valid if params['split'] == 'val' else test
    data_iter = np.random.permutation(len(targ_split))

    if len(params['withExtMask']) and (params['extmask_type'] == 'mask'):
        gt_mask_data = get_dataset('','', params['mask_size'], params['mask_size'],
                params['dataset'] if params['extMask_source']=='gt' else params['extMask_source'],
                params['split'],select_attrs=configs[0]['selected_attrs'], bboxLoader=0, loadMasks = True)
    if len(params['sort_by']):
        resFiles = [json.load(open(fil,'r')) for fil in params['sort_by']]
        for i in xrange(len(resFiles)):
            #if params['sort_score'] not in resFiles[i]['images'][resFiles[i]['images'].keys()[0]]['overall']:
            for k in resFiles[i]['images']:
                img = resFiles[i]['images'][k]
                if 'overall' in resFiles[i]['images'][k]:
                    resFiles[i]['images'][k]['overall'][params['sort_score']] = np.mean([img['perclass'][cls][params['sort_score']] for cls in img['perclass']])
                else:
                    resFiles[i]['images'][k]['overall'] = {}
                    resFiles[i]['images'][k]['overall'][params['sort_score']] = np.mean([img['perclass'][cls][params['sort_score']] for cls in img['perclass']])
        idToScore = {int(k):resFiles[0]['images'][k]['overall'][params['sort_score']] for k in resFiles[0]['images']}
        idToScore = OrderedDict(reversed(sorted(idToScore.items(), key=lambda t: t[1])))
        cocoIdToindex = {v:i for i,v in enumerate(dataset.valid_ids)}
        data_iter = [cocoIdToindex[k] for k in idToScore]
        dataIt2id = {cocoIdToindex[k]:str(k) for k in idToScore}

    if len(params['show_ids'])> 0:
        cocoIdToindex = {v:i for i,v in enumerate(dataset.valid_ids)}
        data_iter = [cocoIdToindex[k] for k in params['show_ids']]

    print len(data_iter)

    print('-----------------------------------------')
    print('%s'%(' | '.join(targ_split.selected_attrs)))
    print('-----------------------------------------')

    flatten = lambda l: [item for sublist in l for item in sublist]

    if params['showreconst'] and len(params['names'])>0:
        params['names'] = flatten([[nm,nm+'-R'] for nm in params['names']])

    #discriminator.load_state_dict(cv['discriminator_state_dict'])
    c_idx = 0
    np.set_printoptions(precision=2)
    padimg = np.zeros((params['image_size'],5,3),dtype=np.uint8)
    padimg[:,:,:] = 128
    if params['showperceptionloss']:
        vggLoss = VGGLoss(network='squeeze')
    cimg_cnt = 0
    mean_hist = [[],[],[]]
    max_hist = [[],[],[]]
    lengths_hist = [[],[],[]]
    if len(params['n_iter']) == 0:
        params['n_iter'] = [0]*len(params['model'])
    while True:
        cimg_cnt+=1
        #import ipdb; ipdb.set_trace()
        idx = data_iter[c_idx]
        x, real_label, boxImg, boxlabel, mask, bbox, curCls  = targ_split[data_iter[c_idx]]
        fp = [targ_split.getfilename(data_iter[c_idx])]

        #if configs[0]['use_gtmask_inp']:
        #    mask = mask[1:,::]

        x = x[None,::]; boxImg = boxImg[None,::]; mask = mask[None,::]; boxlabel = boxlabel[None,::]; real_label = real_label[None,::]

        x, boxImg, mask, boxlabel = solvers[0].to_var(x, volatile=True), solvers[0].to_var(boxImg, volatile=True), solvers[0].to_var(mask, volatile=True), solvers[0].to_var(boxlabel, volatile=True)
        real_label = solvers[0].to_var(real_label, volatile=True)

        fake_image_list = [x]
        if params['showmask']:
            mask_image_list = [x-x]
        else:
            fake_image_list.append(x*(1-mask)+mask)

        deformList = [[], []]
        if len(real_label[0,:].nonzero()):
            #rand_idx = random.choice(real_label[0,:].nonzero()).data[0]
            rand_idx = curCls[0]
            print configs[0]['selected_attrs'][rand_idx]
            if len(params['withExtMask']):
                cocoid = targ_split.getcocoid(idx)
                if params['extmask_type'] == 'mask':
                    mask = solvers[0].to_var(gt_mask_data.getbyIdAndclass(cocoid,configs[0]['selected_attrs'][rand_idx])[None,::], volatile=True)
                elif params['extmask_type'] == 'box':
                    mask = solvers[0].to_var(dataset.getGTMaskInp(idx,configs[0]['selected_attrs'][rand_idx], mask_type=2)[None,::],volatile=True)
                elif params['extmask_type'] == 'randbox':
                    # Nothing to do here, mask is already set to random boxes
                    None
        else:
            rand_idx = curCls[0]
        if params['showdiff']:
            diff_image_list = [x-x] if params['showmask'] else [x-x, x-x]
        for i in xrange(len(params['model'])):
            if configs[i]['use_gtmask_inp']:
                mask = solvers[0].to_var(targ_split.getGTMaskInp(idx, configs[0]['selected_attrs'][rand_idx], mask_type = configs[i]['use_gtmask_inp'])[None,::], volatile=True)
            if len(params['withExtMask']) or params['no_maskgen']:
                withGTMask =  True if params['no_maskgen'] else params['withExtMask'][i]
            else:
                withGTMask = False

            if configs[i]['train_boxreconst']==3:
                mask_target = torch.zeros_like(real_label)
                if len(real_label[0,:].nonzero()):
                    mask_target[0,rand_idx] = 1
                # This variable informs to the mask generator, which class to generate for
                boxlabelInp = boxlabel

            elif configs[i]['train_boxreconst']==2:
                boxlabelfake = torch.zeros_like(boxlabel)
                if configs[i]['use_box_label'] == 2:
                    boxlabelInp = torch.cat([boxlabel, boxlabelfake],dim=1)
                    if params['showreconst']:
                        boxlabelInpRec = torch.cat([boxlabelfake, boxlabel],dim=1)
                mask_target = real_label
            else:
                boxlabelInp = boxlabel
                mask_target = real_label
            if params['showdeform']:
                img, maskOut, deform = solvers[i].forward_generator(x, boxImg=boxImg, mask=mask, imagelabel = mask_target,
                                                                 boxlabel=boxlabelInp, get_feat= True, mask_threshold=params['mask_threshold'],
                                                                 withGTMask=withGTMask, dilate = dilateWeightAll[i],n_iter = params['n_iter'][i])
                fake_image_list.append(img)
                deformList.append(deform)
            else:
                img, maskOut  = solvers[i].forward_generator(x, boxImg=boxImg, mask=mask, imagelabel = mask_target, boxlabel=boxlabelInp,
                                                          mask_threshold=params['mask_threshold'], withGTMask=withGTMask,
                                                          dilate = dilateWeightAll[i],n_iter = params['n_iter'][i])
                fake_image_list.append(img)
            if params['showmask']:
                mask_image_list.append(solvers[i].getImageSizeMask(maskOut)[:,[0,0,0],::])
            if params['showdiff']:
                diff_image_list.append(x-fake_image_list[-1])
            if params['showreconst']:
                if params['showdeform']:
                    img, maskOut, deform = solvers[i].forward_generator(fake_image_list[-1], boxImg=boxImg, mask=mask, imagelabel = mask_target,
                                                                     boxlabel=boxlabelInp, get_feat= True, mask_threshold=params['mask_threshold'],
                                                                     withGTMask=withGTMask, dilate = dilateWeightAll[i], n_iter = params['n_iter'][i])
                    fake_image_list.append(img)
                    deformList.append(deform)
                else:
                    img, maskOut  = solvers[i].forward_generator(fake_image_list[-1], boxImg=boxImg, mask=mask, imagelabel = mask_target,
                                                              boxlabel=boxlabelInp, mask_threshold=params['mask_threshold'], withGTMask=withGTMask,
                                                              dilate = dilateWeightAll[i], n_iter = params['n_iter'][i])
                    fake_image_list.append(img)
                if params['showdiff']:
                    diff_image_list.append(x-fake_image_list[-1])


        if not params['compute_deform_stats']:
            img = make_image(fake_image_list, padimg)
            if params['showdeform']:
                defImg = make_image_with_deform(fake_image_list, deformList, np.vstack([padimg,padimg, padimg]))
                img = np.vstack([img, defImg])
            if params['showmask']:
                imgmask = make_image(mask_image_list,padimg)
                img = np.vstack([img, imgmask])
            if params['showdiff']:
                imgdiff = make_image(diff_image_list,padimg)
                img = np.vstack([img, imgdiff])
            if len(params['names']) > 0:
                nameList =  ['Input']+params['names'] if params['showmask'] else  ['Input', 'Masked Input']+params['names']
                imgNames = np.hstack(flatten([[make_image_with_text((32,x.size(3), 3), nm), padimg[:32,:,:].astype(np.uint8)] for nm in nameList]))
                img = np.vstack([imgNames, img])
            if len(params['sort_by']):
                clsname = configs[0]['selected_attrs'][rand_idx]
                cocoid = dataIt2id[data_iter[c_idx]]
                curr_class_iou = [resFiles[i]['images'][cocoid]['real_scores'][rand_idx]] + [resFiles[i]['images'][cocoid]['perclass'][clsname][params['sort_score']] for i in xrange(len(params['model']))]
                if params['showperceptionloss']:
                    textToPrint = ['P:%.2f, S:%.1f'%(vggLoss(fake_image_list[0], fake_image_list[i]).data[0],curr_class_iou[i]) for i in xrange(len(fake_image_list))]
                else:
                    textToPrint = ['S:%.1f'%(curr_class_iou[i]) for i in xrange(len(fake_image_list))]
                if len(params['show_also']):
                    # Additional data to print
                    for val in params['show_also']:
                        curval = [0.] + [resFiles[i]['images'][cocoid]['perclass'][clsname][val][rand_idx] for i in xrange(len(params['model']))]
                        textToPrint = [txt + ' %s:%.1f'%(val[0], curval[i]) for i,txt in enumerate(textToPrint)]

                imgScore =  np.hstack(flatten([[make_image_with_text((32,x.size(3), 3),
                                      textToPrint[i]),
                                      padimg[:32,:,:].astype(np.uint8)] for i in xrange(len(fake_image_list))]))
                img = np.vstack([img, imgScore])
            elif params['showperceptionloss']:
                imgScore = np.hstack(flatten([[make_image_with_text((32,x.size(3), 3), '%.2f'%vggLoss(fake_image_list[0],fake_image_list[i]).data[0]), padimg[:32,:,:].astype(np.uint8)] for i in xrange(len(fake_image_list))]))
                img = np.vstack([img, imgScore])


            #if params['showmask']:
            #    imgmask = make_image(mask_list)
            #    img = np.vstack([img, imgmask])
            #if params['compmodel']:
            #    imgcomp = make_image(fake_image_list_comp)
            #    img = np.vstack([img, imgcomp])
            #    if params['showdiff']:
            #        imgdiffcomp = make_image([fimg - fake_image_list_comp[0] for fimg in fake_image_list_comp])
            #        img = np.vstack([img, imgdiffcomp])
            cv2.imshow('frame',img if params['scaleDisp']==0 else cv2.resize(img,None, fx = params['scaleDisp'], fy=params['scaleDisp']))
            keyInp = cv2.waitKey(0)

            if keyInp & 0xFF == ord('q'):
                break
            elif keyInp & 0xFF == ord('b'):
                #print keyInp & 0xFF
                c_idx = c_idx-1
            elif (keyInp & 0xFF == ord('s')):
                #sample_dir = join(params['sample_dump_dir'], basename(params['model'][0]).split('.')[0])
                sample_dir = join(params['sample_dump_dir'],'_'.join([params['split']]+params['names']))
                if not exists(sample_dir):
                    makedirs(sample_dir)
                fnames = ['%s.png' % splitext(basename(f))[0] for f in fp]
                fpaths = [join(sample_dir, f) for f in fnames]
                imgSaveName = fpaths[0]
                if params['savesepimages']:
                    saveIndividImages(fake_image_list, mask_image_list, nameList, sample_dir, fp, configs[0]['selected_attrs'][rand_idx])
                else:
                    print 'Saving into file: ' + imgSaveName
                    cv2.imwrite(imgSaveName, img)
                c_idx += 1
            else:
                c_idx += 1
        else:
            for di in xrange(len(deformList)):
                if len(deformList[di])>0 and len(deformList[di][0])>0:
                    for dLidx,d in enumerate(deformList[di]):
                        lengths, mean, maxl = compute_deform_statistics(d[1], d[0])
                        mean_hist[dLidx].append(mean)
                        max_hist[dLidx].append(maxl)
                        lengthsH = np.histogram(lengths, bins=np.arange(0,128,0.5))[0]
                        if lengths_hist[dLidx] == []:
                            lengths_hist[dLidx] = lengthsH
                        else:
                            lengths_hist[dLidx] += lengthsH

        if params['compute_deform_stats'] and (cimg_cnt < params['compute_deform_stats']):
            print np.mean(mean_hist[0])
            print np.mean(mean_hist[1])
            print np.mean(mean_hist[2])
            print np.mean(max_hist[0])
            print np.mean(max_hist[1])
            print np.mean(max_hist[2])

            print lengths_hist[0]
            print lengths_hist[1]
            print lengths_hist[2]
            break

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--showdiff', type=int, default=0)
  parser.add_argument('--showperceptionloss', type=int, default=0)
  parser.add_argument('--showdeform', type=int, default=0)
  parser.add_argument('--showmask', type=int, default=0)
  #parser.add_argument('--showclassifier', type=int, default=0)
  parser.add_argument('--showreconst', type=int, default=0)
  parser.add_argument('-d', '--dataset', dest='dataset',  type=str, default='coco', help='dataset: celeb')
  parser.add_argument('-m', '--model', type=str, default=[], nargs='+', help='checkpoint to resume training from')
  parser.add_argument('-n', '--names', type=str, default=[], nargs='+', help='checkpoint to resume training from')
  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=1, help='max batch size')
  parser.add_argument('--sample_dump_dir', type=str, default='gen_samples', help='print every x iters')
  parser.add_argument('--swap_attr', type=str, default='rand', help='which attribute to swap')
  parser.add_argument('--split', type=str, default='val', help='which attribute to swap')
  parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')

  parser.add_argument('--sort_by', type=str, default=[], nargs='+', help='Evaluation scores to visualize')
  parser.add_argument('--sort_score', type=str, default='iou', help='Evaluation scores to visualize')
  parser.add_argument('--show_also', type=str, nargs = '+', default=[], help='Evaluation scores to visualize')
  parser.add_argument('--use_same_g', type=str, default=[], nargs='+', help='Evaluation scores to visualize')

  # Deformations applied to mnist images;
  parser.add_argument('--no_maskgen', type=int, default=0)
  parser.add_argument('--randomrotate', type=int, default=90)
  parser.add_argument('--randomscale', type=float, nargs='+', default=[0.5,0.5])
  parser.add_argument('--image_size', type=int, default=128)
  parser.add_argument('--scaleDisp', type=int, default=0)
  parser.add_argument('--box_size', type=int, default=64)
  parser.add_argument('--mask_threshold', type=float, default=0.3)
  parser.add_argument('--withExtMask', type=int, nargs ='+', default=[])
  parser.add_argument('--extmask_type', type=str, default='mask')
  parser.add_argument('--n_iter', type=int, nargs ='+', default=[])
  parser.add_argument('--mask_size', type=int, default=32)
  parser.add_argument('--dilateMask', type=int, default=[], nargs='+')
  parser.add_argument('--datafile', type=str, default='datasetBoxAnn_80pcMaxObj.json')
  parser.add_argument('--extMask_source', type=str, default='gt')
  parser.add_argument('--n_boxes', type=int, default=4)
  parser.add_argument('--show_ids', type=int, default=[], nargs='+')

  parser.add_argument('--savesepimages', type=int, default=0)
  parser.add_argument('--filter_by_mincooccur', type=float, default=-1.)
  parser.add_argument('--only_indiv_occur', type=float, default=0)


  parser.add_argument('--compute_deform_stats', type=int, default=0)
  parser.add_argument('--max_object_size', type=float, default=0.3)

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  params['cuda'] = not args.no_cuda
  print json.dumps(params, indent = 2)
  gen_samples(params)
