import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as FN
from torchvision.datasets import ImageFolder, MNIST
from PIL import Image
import json
import numpy as np
import random
from utils import computeIOU, computeContainment, computeUnionArea
from pycocotools.coco import COCO as COCOTool
from collections import defaultdict
from random import shuffle
from copy import copy

class CocoDatasetBBoxSample(Dataset):
    def __init__(self, transform, mode, select_attrs=[], datafile='datasetBoxAnn.json', out_img_size=128, bbox_out_size=64,
                 balance_classes=0, onlyrandBoxes=False, max_object_size=0., max_with_union=True, use_gt_mask=False,
                 boxrotate=0, n_boxes = 1, square_resize=0, filter_by_mincooccur = -1., only_indiv_occur = 0., augmenter_mode=0):
        self.image_path = os.path.join('data','coco','images')
        self.transform = transform
        self.mode = mode
        self.n_boxes = n_boxes
        self.iouThresh = 0.5
        self.dataset = json.load(open(os.path.join('data','coco',datafile),'r'))
        self.num_data = len(self.dataset['images'])
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.out_img_size = out_img_size
        self.square_resize = square_resize
        self.bbox_out_size = bbox_out_size
        self.filter_by_mincooccur = filter_by_mincooccur
        self.only_indiv_occur = only_indiv_occur
        #self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.selected_attrs = select_attrs
        if len(select_attrs) == 0:
            self.selected_attrs = [attr['name'] for attr in self.dataset['categories']]
        self.balance_classes = balance_classes
        self.onlyrandBoxes = onlyrandBoxes
        self.max_object_size = max_object_size
        self.max_with_union= max_with_union
        self.use_gt_mask = use_gt_mask
        self.boxrotate = boxrotate
        self.augmenter_mode = augmenter_mode
        if self.boxrotate:
            self.rotateTrans = transforms.Compose([transforms.RandomRotation(boxrotate,resample=Image.NEAREST)])
        if use_gt_mask == 1:
            self.mask_trans =transforms.Compose([transforms.Resize(out_img_size if not square_resize else [out_img_size, out_img_size] , interpolation=Image.NEAREST), transforms.CenterCrop(out_img_size)])
            self.mask_provider = CocoMaskDataset(self.mask_trans, mode, select_attrs=self.selected_attrs, balance_classes=balance_classes)

        self.randHFlip = 'Flip' in transform

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')
        self.imgId2idx = {imid:i for i,imid in enumerate(self.valid_ids)}

        self.num_data = len(self.dataset['images'])

    def preprocess(self):
        for i, attr in enumerate(self.dataset['categories']):
            self.attr2idx[attr['name']] = i
            self.idx2attr[i] = attr['name']
            self.catid2attr[attr['id']] = attr['name']

        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}

        # First remove unwanted splits:
        self.dataset['images'] = [img for img in self.dataset['images'] if img['split'] == self.mode]
        if self.max_object_size > 0.:
            validImgs = []
            for img in self.dataset['images']:
                if not self.max_with_union:
                    maxSize = max([bb['bbox'][2]*bb['bbox'][3] for bb in img['bboxAnn']])
                else:
                    boxByCls = defaultdict(list)
                    for bb in img['bboxAnn']:
                        boxByCls[bb['cid']].append(bb['bbox'])
                    unionAreas = [computeUnionArea(boxes) for cid,boxes in boxByCls.iteritems()]
                    maxSize = max(unionAreas)
                if maxSize < self.max_object_size:
                    validImgs.append(img)
            print ' %d of %d images left after size filtering'%(len(validImgs), len(self.dataset['images']))
            self.dataset['images'] = validImgs

        self.valid_ids = [img['cocoid'] for img in self.dataset['images']]
        self.catsInImg = {}

        selset = set(self.selected_attrs)
        for i, img in enumerate(self.dataset['images']):
            self.dataset['images'][i]['label'] = np.zeros(max(len(selset),1))
            self.dataset['images'][i]['bboxAnn'] = [bb for bb in img['bboxAnn'] if self.catid2attr[bb['cid']] in selset]

            # Correct BBox for Resize(of smaller edge) and CenterCrop
            fixedbbox = []
            imgSize = self.dataset['images'][i]['imgSize']
            maxSide = np.argmax(imgSize)
            for j in xrange(len(self.dataset['images'][i]['bboxAnn'])):
                cbbox = self.dataset['images'][i]['bboxAnn'][j]
                maxSideLen = int(float(self.out_img_size * imgSize[maxSide]) / (imgSize[1-maxSide])) if not self.square_resize else self.out_img_size
                assert(maxSideLen >= self.out_img_size)
                newStartCord = round((maxSideLen - self.out_img_size)/2.)
                boxStart = min( max(cbbox['bbox'][maxSide]*maxSideLen - newStartCord, 0),  self.out_img_size)
                boxEnd =  min(max((cbbox['bbox'][maxSide]+cbbox['bbox'][maxSide+2])*maxSideLen - newStartCord, 0), self.out_img_size)
                length = boxEnd - boxStart
                if length >= 1:
                    cbbox['bbox'][maxSide] = float(boxStart)/self.out_img_size
                    cbbox['bbox'][maxSide+2] = float(length)/self.out_img_size
                    if cbbox['bbox'][1-maxSide+2] >= 1./self.out_img_size:
                        fixedbbox.append(cbbox)
                        if cbbox['bbox'][0]<0. or cbbox['bbox'][1] < 0. or cbbox['bbox'][0]>1.0 or cbbox['bbox'][1]> 1.0:
                            import ipdb; ipdb.set_trace()
            self.dataset['images'][i]['bboxAnn'] = fixedbbox
            self.dataset['images'][i]['label'][[self.sattr_to_idx[self.catid2attr[bb['cid']]] for bb in img['bboxAnn']]] = 1.

            # Convert bbox data to numpy arrays
            #for j, bb in enumerate(self.dataset['images'][i]['bboxAnn']):
            #    self.dataset['images'][i]['bboxAnn'][j]['bbox'] = np.array(bb['bbox'])
            # Create bbox labels.
            if self.augmenter_mode:
                lab_in_img = img['label'].nonzero()[0]
                self.dataset['images'][i]['label_seq'] = lab_in_img
                n_lab_in_img = len(lab_in_img)
                #self.dataset['images'][i]['cls_affect']  = np.zeros((n_lab_in_img,n_lab_in_img))
                idx2aidx = {l:li for li,l in enumerate(lab_in_img)}
                boxByCls = defaultdict(list)
                for bb in img['bboxAnn']:
                    boxByCls[idx2aidx[self.sattr_to_idx[self.catid2attr[bb['cid']]]]].append(bb['bbox'])
                self.dataset['images'][i]['cls_affect'] = np.array([[min([max([computeContainment(bb1, bb2)[0] for bb1 in boxByCls[li1]]) for bb2 in boxByCls[li2]]) for li2 in xrange(n_lab_in_img)] for li1 in xrange(n_lab_in_img)])

            for j, bb in enumerate(self.dataset['images'][i]['bboxAnn']):
                #Check for IOU > 0.5 with other bbox
                iouAr = [computeContainment(bb['bbox'], bother['bbox'])[0] for bother in self.dataset['images'][i]['bboxAnn']]
                self.dataset['images'][i]['bboxAnn'][j]['box_label'] = np.zeros(len(selset))
                self.dataset['images'][i]['bboxAnn'][j]['box_label'][[self.sattr_to_idx[self.catid2attr[self.dataset['images'][i]['bboxAnn'][ii]['cid']]] for ii,iv in enumerate(iouAr) if iv>self.iouThresh]] = 1.

        if self.filter_by_mincooccur >= 0. or self.only_indiv_occur:
            clsToSingleOccur = defaultdict(list)
            clsCounts = np.zeros(len(self.selected_attrs))
            clsIndivCounts = np.zeros(len(self.selected_attrs))
            for i, img in enumerate(self.dataset['images']):
                imgCls = set()
                for bb in img['bboxAnn']:
                    imgCls.add(self.catid2attr[bb['cid']])
                imgCls = list(imgCls)
                if len(imgCls)==1:
                    clsIndivCounts[self.sattr_to_idx[imgCls[0]]] += 1.
                    clsToSingleOccur[imgCls[0]].append(i)
                else:
                    clsCounts[[self.sattr_to_idx[cls] for cls in imgCls]] += 1.

            if self.filter_by_mincooccur >= 0.:
                n_rem_counts = clsIndivCounts - self.filter_by_mincooccur/(1-self.filter_by_mincooccur) * clsCounts
                allRemIds = set()
                for cls in self.selected_attrs:
                    if n_rem_counts[self.sattr_to_idx[cls]] > 0:
                        n_rem_idx = np.arange(len(clsToSingleOccur[cls]))
                        np.random.shuffle(n_rem_idx)
                        n_rem_idx = n_rem_idx[:int(n_rem_counts[self.sattr_to_idx[cls]])]
                        allRemIds.update([clsToSingleOccur[cls][ri] for ri in n_rem_idx])

                self.dataset['images'] = [img for i,img in enumerate(self.dataset['images']) if i not in allRemIds]
            elif self.only_indiv_occur:
                allKeepIds = set()
                for cls in self.selected_attrs:
                    allKeepIds.update(clsToSingleOccur[cls])

                self.dataset['images'] = [img for i,img in enumerate(self.dataset['images']) if i in allKeepIds]

            self.valid_ids = [img['cocoid'] for img in self.dataset['images']]
            print ' %d images left after co_occurence filtering'%(len(self.valid_ids))

        self.attToImgId = defaultdict(set)
        for i, img in enumerate(self.dataset['images']):
            classesInImg = [self.catid2attr[bb['cid']] for bb in img['bboxAnn'] if self.catid2attr[bb['cid']] in selset]
            if len(classesInImg):
                self.catsInImg[i] = classesInImg
                for att in classesInImg:
                    self.attToImgId[att].add(i)
            else:
                self.attToImgId['bg'].add(i)
                self.catsInImg[i] = ['bg']
        self.attToImgId = {k:list(v) for k,v in self.attToImgId.iteritems()}


    def randomBBoxSample(self, index, max_area = -1):
        # With 50% chance sample from background or foreground
        # Minimum size
        minLen = 0.1
        maxLen = 0.7
        maxIou = 0.3
        cbboxList = self.dataset['images'][index]['bboxAnn'] if not self.onlyrandBoxes else []
        n_t = 0
        while 1:
            if len(cbboxList) and (random.random()<0.9):
                cbid = random.randrange(len(cbboxList))
                sbox = self.dataset['images'][index]['bboxAnn'][cbid]
                return copy(sbox['bbox']),sbox['box_label'], cbid
            else:
                # sample a random background box
                cbid = None
                tL_x, tL_y = random.uniform(0,1.-minLen-0.01), random.uniform(0,1.-minLen-0.01)
                l_x = random.uniform(minLen, min(1.-tL_x,maxLen))
                l_y = random.uniform(minLen, min(1.-tL_y,maxLen))
                sbox = [tL_x, tL_y, l_x, l_y]
                # Prepare label for this box
                bboxLabel = np.zeros(max(len(self.selected_attrs),1))
                # Test for overlap with foreground objects
                noOverlap = True
                #if len(cbboxList):
                for bb in cbboxList:
                    iou, aInb, bIna = computeIOU(sbox, bb['bbox'])
                    if iou > maxIou or aInb >0.8:
                        noOverlap = False
                    if bIna > 0.8:
                        bboxLabel[self.sattr_to_idx[self.catid2attr[bb['cid']]]] = 1
                if noOverlap and ((max_area < 0) or ((sbox[2]*sbox[3])< max_area) or (n_t>5)):
                    return sbox, bboxLabel, cbid
            n_t += 1

    def __getitem__(self, index):
        # In this situation ignore index and sample classes uniformly
        if self.balance_classes==1:
            currCls = random.choice(self.attToImgId.keys())
            index = random.choice(self.attToImgId[currCls])
        elif self.balance_classes==2:
            currCls = random.choice(self.catsInImg[index]) if ('person' not in self.catsInImg[index]) or (random.rand()<0.2) else 'person'
        else:
            currCls = random.choice(self.catsInImg[index])

        cid = [self.sattr_to_idx[currCls]] if currCls != 'bg' else [0]

        if not self.augmenter_mode:
            returnvals = self.getbyIndexAndclass(index, cid)
        else:
            returnvals = self.getbyIndexAndclassAugmentMode(index)

        return tuple(returnvals)

    def getbyIdAndclass(self, imgid, cls, hflip=0):
        index = self.imgId2idx[imgid]
        cid = [self.sattr_to_idx[cls]] if cls != 'bg' else [0]
        returnvals = self.getbyIndexAndclass(index, cid)
        return tuple(returnvals)

    def getbyIndexAndclass(self, index, cid):

        image = Image.open(os.path.join(self.image_path,self.dataset['images'][index]['filepath'], self.dataset['images'][index]['filename']))
        currCls = self.selected_attrs[cid[0]]
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')

        sampbbox, bboxLabel, cbid = self.randomBBoxSample(index, 0.5)
        extra_boxes = []
        if self.n_boxes > 1:
            # Sample random number of boxes between 1 and n_boxes
            c_nbox = np.random.randint(0,self.n_boxes)
            c_area = sampbbox[2]*sampbbox[3]
            for i in xrange(c_nbox):
                # Also stop at total area > 50%
                if c_area < 0.5:
                    bsamp, _, _ = self.randomBBoxSample(index, 0.6-c_area) # Extra 10% to make the sampling easier
                    extra_boxes.append(bsamp)
                    c_area += bsamp[2]*bsamp[3]
                else:
                    break

        label = self.dataset['images'][index]['label']

        # Apply transforms to the image.
        image = self.transform[0](image)
        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            sampbbox[0] = 1.0-(sampbbox[0]+sampbbox[2])
        if self.use_gt_mask==1:
            # Use GT masks as input
            gtMask = self.mask_provider.getbyIdAndclass(self.dataset['images'][index]['cocoid'], currCls, hflip=hflip)
        elif self.use_gt_mask==2:
            # Use GT boxes as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== currCls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                gtMask[0,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1.
        elif self.use_gt_mask==3:
            # Use GT centerpoints as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== currCls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                cent = [bbox[0] + bbox[2]//2, bbox[1]+bbox[3]//2]
                # center is marked by a 3x3 square patch
                gtMask[0,cent[1]-1:cent[1]+2,cent[0]-1:cent[0]+2] = 1.


        #Convert BBox to actual co-ordinates
        sampbbox = [int(bc*self.out_img_size) for bc in sampbbox]
        boxCrop = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
        # Create Mask
        mask = torch.zeros(1,self.out_img_size,self.out_img_size)
        mask[0,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
        if self.n_boxes > 1 and len(extra_boxes):
            for box in extra_boxes:
                box = [int(bc*self.out_img_size) for bc in box]
                mask[0,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1.

        if self.boxrotate:
            mask = torch.FloatTensor(np.asarray(self.rotateTrans(Image.fromarray(mask.numpy()[0]))))[None,::]
        if self.use_gt_mask:
            mask = torch.cat([mask, gtMask], dim=0)

        return self.transform[-1](image), torch.FloatTensor(label), self.transform[-1](boxCrop), torch.FloatTensor(bboxLabel), mask, torch.IntTensor(sampbbox), torch.LongTensor(cid)

    def getbyIndexAndclassAugmentMode(self, index):

        imgData = self.dataset['images'][index]

        image = Image.open(os.path.join(self.image_path,imgData['filepath'], imgData['filename']))
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')

        gtBoxes = np.zeros((len(self.selected_attrs),4))
        for bbox in imgData['bboxAnn']:
            gtBoxes[self.sattr_to_idx[self.catid2attr[bbox['cid']]],:] = bbox['bbox']

        label = imgData['label']

        # Apply transforms to the image.
        image = self.transform[0](image)
        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            gtBoxes[np.array(label,dtype=np.int),0] = 1.0-(gtBoxes[np.array(label,dtype=np.int),0]+gtBoxes[np.array(label,dtype=np.int),2])

        #Get class effect;
        class_effect  = np.zeros((len(self.selected_attrs),len(self.selected_attrs)))
        class_effect[np.meshgrid(imgData['label_seq'],imgData['label_seq'])] = imgData['cls_affect']

        #Convert BBox to actual co-ordinates
        return self.transform[-1](image), torch.FloatTensor(label), torch.FloatTensor(gtBoxes), torch.LongTensor([imgData['cocoid']]), torch.LongTensor([hflip]), torch.FloatTensor(class_effect.T)

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        return self.dataset['images'][index]['filename']

    def getfilename_bycocoid(self, cocoid):
        return self.dataset['images'][self.imgId2idx[cocoid]]['filename']

    def getcocoid(self, index):
        return self.dataset['images'][index]['cocoid']

    def getGTMaskInp(self, index, cls, hflip=False, mask_type=None):
        what_mask = self.use_gt_mask if mask_type is None else mask_type
        if what_mask==1:
            # Use GT masks as input
            gtMask = self.mask_provider.getbyIdAndclass(self.dataset['images'][index]['cocoid'], cls, hflip=hflip)
        elif what_mask==2:
            # Use GT boxes as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== cls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                gtMask[0,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1.
        elif what_mask==3:
            # Use GT centerpoints as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== cls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                cent = [bbox[0] + bbox[2]//2, bbox[1]+bbox[3]//2]
                # center is marked by a 3x3 square patch
                gtMask[0,cent[1]-1:cent[1]+2,cent[0]-1:cent[0]+2] = 1.
        else:
            gtMask = None

        return gtMask

class ADE20k(Dataset):
    def __init__(self, transform, split, select_attrs=[], out_img_size=128, bbox_out_size=64,
                 max_object_size=0., max_with_union=True, use_gt_mask=False,
                 boxrotate=0, n_boxes = 1, square_resize=0) :
        self.image_path = os.path.join('data','ade20k')
        self.transform = transform
        self.split = split
        self.n_boxes = n_boxes
        self.iouThresh = 0.5
        datafile = 'train.odgt' if split == 'train' else 'validation.odgt'
        self.datafile = os.path.join('data','ade20k',datafile)
        self.dataset = [json.loads(x.rstrip()) for x in open(self.datafile, 'r')]
        self.num_data = len(self.dataset)
        clsData = open('data/ade20k/object150_info.csv','r').read().splitlines()
        self.clsidx2attr = {i:ln.split(',')[-1] for i, ln in enumerate(clsData[1:])}
        self.clsidx2Stuff = {i:int(ln.split(',')[-2]) for i, ln in enumerate(clsData[1:])}
        self.validCatIds = set([i for i in self.clsidx2Stuff if not self.clsidx2Stuff[i]])
        self.maskSample = 'nonStuff'
        self.out_img_size = out_img_size
        self.square_resize = square_resize
        self.bbox_out_size = bbox_out_size
        #self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.selected_attrs = ['background']
        self.max_object_size = max_object_size
        self.max_with_union= max_with_union
        self.use_gt_mask = use_gt_mask
        self.boxrotate = boxrotate
        if self.boxrotate:
            self.rotateTrans = transforms.Compose([transforms.RandomRotation(boxrotate,resample=Image.NEAREST)])
        if use_gt_mask == 1:
            self.mask_transform = transforms.Compose([transforms.Resize(out_img_size if not square_resize else [out_img_size, out_img_size] , interpolation=Image.NEAREST), transforms.CenterCrop(out_img_size)])

        self.valid_ids = []
        for i,img in enumerate(self.dataset):
            imid = int(os.path.basename(img['fpath_img']).split('.')[0].split('_')[-1])
            self.dataset[i]['image_id'] = imid
            self.valid_ids.append(imid)

        self.randHFlip = 'Flip' in transform

        print ('Start preprocessing dataset..!')
        print ('Finished preprocessing dataset..!')
        self.imgId2idx = {imid:i for i,imid in enumerate(self.valid_ids)}

    def randomBBoxSample(self, max_area = -1):
        # With 50% chance sample from background or foreground
        # Minimum size
        minLen = 0.1
        maxLen = 0.7
        maxIou = 0.3
        cbboxList = []
        n_t = 0
        while 1:
            # sample a random background box
            cbid = None
            tL_x, tL_y = random.uniform(0,1.-minLen-0.01), random.uniform(0,1.-minLen-0.01)
            l_x = random.uniform(minLen, min(1.-tL_x,maxLen))
            l_y = random.uniform(minLen, min(1.-tL_y,maxLen))
            sbox = [tL_x, tL_y, l_x, l_y]
            # Prepare label for this box
            bboxLabel = np.zeros(max(len(self.selected_attrs),1))
            #if len(cbboxList):
            if ((max_area < 0) or ((sbox[2]*sbox[3])< max_area) or (n_t>5)):
                return sbox, bboxLabel, cbid
            n_t += 1

    def __getitem__(self, index):
        # In this situation ignore index and sample classes uniformly
        returnvals = self.getbyIndexAndclass(index)

        return tuple(returnvals)

    def getbyIdAndclass(self, imgid, cls, hflip=0):
        index = self.imgId2idx[imgid]
        cid = [self.sattr_to_idx[cls]] if cls != 'bg' else [0]
        returnvals = self.getbyIndexAndclass(index, cid)
        return tuple(returnvals)

    def getbyIndexAndclass(self, index,cls=None):

        imgDb = self.dataset[index]
        image_id = imgDb['image_id']
        image = Image.open(os.path.join(self.image_path,imgDb['fpath_img']))
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')
        cid = [0]

        sampbbox, bboxLabel, cbid = self.randomBBoxSample(0.5)
        extra_boxes = []
        if self.n_boxes > 1:
            # Sample random number of boxes between 1 and n_boxes
            c_nbox = np.random.randint(0,self.n_boxes)
            c_area = sampbbox[2]*sampbbox[3]
            for i in xrange(c_nbox):
                # Also stop at total area > 50%
                if c_area < 0.5:
                    bsamp, _, _ = self.randomBBoxSample(0.6-c_area) # Extra 10% to make the sampling easier
                    extra_boxes.append(bsamp)
                    c_area += bsamp[2]*bsamp[3]
                else:
                    break

        label = np.ones(max(len(self.selected_attrs),1))

        # Apply transforms to the image.
        image = self.transform[0](image)
        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            sampbbox[0] = 1.0-(sampbbox[0]+sampbbox[2])
        if self.use_gt_mask==1:
            # Use GT masks as input
            gtMask = self.getGTMaskInp(index, hflip=hflip)

        #Convert BBox to actual co-ordinates
        sampbbox = [int(bc*self.out_img_size) for bc in sampbbox]
        boxCrop = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
        # Create Mask
        mask = torch.zeros(1,self.out_img_size,self.out_img_size)
        mask[0,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
        if self.n_boxes > 1 and len(extra_boxes):
            for box in extra_boxes:
                box = [int(bc*self.out_img_size) for bc in box]
                mask[0,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1.

        if self.boxrotate:
            mask = torch.FloatTensor(np.asarray(self.rotateTrans(Image.fromarray(mask.numpy()[0]))))[None,::]
        if self.use_gt_mask:
            mask = torch.cat([mask, gtMask], dim=0)

        return self.transform[-1](image), torch.FloatTensor(label), torch.FloatTensor(bboxLabel), torch.FloatTensor(bboxLabel), mask, torch.IntTensor(sampbbox), torch.LongTensor(cid)

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        return os.path.basename(self.dataset[index]['fpath_img'])

    def getfilename_bycocoid(self, cocoid):
        return os.path.basename(self.dataset[self.imgId2idx[cocoid]]['fpath_img'])

    def getcocoid(self, index):
        return self.dataset[index]['image_id']

    def getGTMaskInp(self, index, cls=None, hflip=False, mask_type=None):
        imgDb = self.dataset[index]
        segmImg = np.array(Image.open(os.path.join(self.image_path,imgDb['fpath_segm'])))-1
        presentClass = np.unique(segmImg)
        validClass = map(lambda x: x in self.validCatIds, presentClass)
        chosenIdx = np.random.choice(presentClass[validClass]) if np.sum(validClass) > 0 else -10
        if chosenIdx < 0:
            maskTotal = np.zeros((self.out_img_size,self.out_img_size))
            sampbbox, bboxLabel, cbid = self.randomBBoxSample(0.5)
            sampbbox = [int(bc*self.out_img_size) for bc in sampbbox]
            maskTotal[sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
        else:
            maskTotal = (segmImg == chosenIdx).astype(np.float)
        if hflip:
            maskTotal = maskTotal[:,::-1]

        mask = torch.FloatTensor(np.asarray(self.mask_transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]

        return mask


class BelgaLogoBBoxSample(Dataset):
    def __init__(self, transform, mode, select_attrs=[], datafile='dataset.json', out_img_size=128, bbox_out_size=64,
                 balance_classes=0, onlyrandBoxes=False, max_object_size=0., max_with_union=True, use_gt_mask=False,
                 boxrotate=0, n_boxes = 1):
        self.image_path = os.path.join('data','belgalogos','images')
        self.transform = transform
        self.mode = mode
        self.n_boxes = n_boxes
        self.iouThresh = 0.5
        self.dataset = json.load(open(os.path.join('data','belgalogos',datafile),'r'))
        self.num_data = len(self.dataset['images'])
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.out_img_size = out_img_size
        self.bbox_out_size = bbox_out_size
        #self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.selected_attrs = select_attrs
        self.balance_classes = balance_classes
        self.onlyrandBoxes = onlyrandBoxes
        self.max_object_size = max_object_size
        self.max_with_union= max_with_union
        self.use_gt_mask = use_gt_mask
        self.boxrotate = boxrotate
        if self.boxrotate:
            self.rotateTrans = transforms.Compose([transforms.RandomRotation(boxrotate,resample=Image.NEAREST)])
        if use_gt_mask == 1:
            print ' Not Supported'
            assert(0)

        self.randHFlip = 'Flip' in transform

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')
        self.imgId2idx = {imid:i for i,imid in enumerate(self.valid_ids)}

        self.num_data = len(self.dataset['images'])

    def preprocess(self):
        for i, attr in enumerate(self.dataset['categories']):
            self.attr2idx[attr['name']] = i
            self.idx2attr[i] = attr['name']
            self.catid2attr[attr['id']] = attr['name']

        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}

        # First remove unwanted splits:
        self.dataset['images'] = [img for img in self.dataset['images'] if img['split'] == self.mode]
        if self.max_object_size > 0.:
            validImgs = []
            for img in self.dataset['images']:
                if not self.max_with_union:
                    maxSize = max([bb['bbox'][2]*bb['bbox'][3] for bb in img['bboxAnn']])
                else:
                    boxByCls = defaultdict(list)
                    for bb in img['bboxAnn']:
                        boxByCls[bb['cid']].append(bb['bbox'])
                    unionAreas = [computeUnionArea(boxes) for cid,boxes in boxByCls.iteritems()]
                    maxSize = max(unionAreas)
                if maxSize < self.max_object_size:
                    validImgs.append(img)
            print ' %d of %d images left after size filtering'%(len(validImgs), len(self.dataset['images']))
            self.dataset['images'] = validImgs

        self.valid_ids = [img['id'] for img in self.dataset['images']]
        self.catsInImg = {}

        selset = set(self.selected_attrs)
        for i, img in enumerate(self.dataset['images']):
            self.dataset['images'][i]['label'] = np.zeros(max(len(selset),1))
            self.dataset['images'][i]['bboxAnn'] = [bb for bb in img['bboxAnn'] if self.catid2attr[bb['cid']] in selset]

            # Correct BBox for Resize(of smaller edge) and CenterCrop
            fixedbbox = []
            imgSize = self.dataset['images'][i]['imgSize']
            maxSide = np.argmax(imgSize)
            for j in xrange(len(self.dataset['images'][i]['bboxAnn'])):
                cbbox = self.dataset['images'][i]['bboxAnn'][j]
                maxSideLen = int(float(self.out_img_size * imgSize[maxSide]) / (imgSize[1-maxSide]))
                assert(maxSideLen >= self.out_img_size)
                newStartCord = round((maxSideLen - self.out_img_size)/2.)
                boxStart = min( max(cbbox['bbox'][maxSide]*maxSideLen - newStartCord, 0),  self.out_img_size)
                boxEnd =  min(max((cbbox['bbox'][maxSide]+cbbox['bbox'][maxSide+2])*maxSideLen - newStartCord, 0), self.out_img_size)
                length = boxEnd - boxStart
                if length > 5:
                    cbbox['bbox'][maxSide] = float(boxStart)/self.out_img_size
                    cbbox['bbox'][maxSide+2] = float(length)/self.out_img_size
                    if cbbox['bbox'][1-maxSide+2] >= 0.04:
                        fixedbbox.append(cbbox)
                        if cbbox['bbox'][0]<0. or cbbox['bbox'][1] < 0. or cbbox['bbox'][0]>1.0 or cbbox['bbox'][1]> 1.0:
                            import ipdb; ipdb.set_trace()
            self.dataset['images'][i]['bboxAnn'] = fixedbbox
            self.dataset['images'][i]['label'][[self.sattr_to_idx[self.catid2attr[bb['cid']]] for bb in img['bboxAnn']]] = 1.

            # Convert bbox data to numpy arrays
            #for j, bb in enumerate(self.dataset['images'][i]['bboxAnn']):
            #    self.dataset['images'][i]['bboxAnn'][j]['bbox'] = np.array(bb['bbox'])
            # Create bbox labels.
            for j, bb in enumerate(self.dataset['images'][i]['bboxAnn']):
                #Check for IOU > 0.5 with other bbox
                iouAr = [computeContainment(bb['bbox'], bother['bbox'])[0] for bother in self.dataset['images'][i]['bboxAnn']]
                self.dataset['images'][i]['bboxAnn'][j]['box_label'] = np.zeros(len(selset))
                self.dataset['images'][i]['bboxAnn'][j]['box_label'][[self.sattr_to_idx[self.catid2attr[self.dataset['images'][i]['bboxAnn'][ii]['cid']]] for ii,iv in enumerate(iouAr) if iv>self.iouThresh]] = 1.

        self.attToImgId = defaultdict(set)
        for i, img in enumerate(self.dataset['images']):
            classesInImg = [self.catid2attr[bb['cid']] for bb in img['bboxAnn'] if self.catid2attr[bb['cid']] in selset]
            if len(classesInImg):
                self.catsInImg[i] = classesInImg
                for att in classesInImg:
                    self.attToImgId[att].add(i)
            else:
                self.attToImgId['bg'].add(i)
                self.catsInImg[i] = ['bg']
        self.attToImgId = {k:list(v) for k,v in self.attToImgId.iteritems()}


    def randomBBoxSample(self, index, max_area = -1):
        # With 50% chance sample from background or foreground
        # Minimum size
        minLen = 0.1
        maxLen = 0.7
        maxIou = 0.3
        cbboxList = self.dataset['images'][index]['bboxAnn'] if not self.onlyrandBoxes else []
        n_t = 0
        while 1:
            if len(cbboxList) and (random.random()<0.9):
                cbid = random.randrange(len(cbboxList))
                sbox = self.dataset['images'][index]['bboxAnn'][cbid]
                return copy(sbox['bbox']),sbox['box_label'], cbid
            else:
                # sample a random background box
                cbid = None
                tL_x, tL_y = random.uniform(0,1.-minLen-0.01), random.uniform(0,1.-minLen-0.01)
                l_x = random.uniform(minLen, min(1.-tL_x,maxLen))
                l_y = random.uniform(minLen, min(1.-tL_y,maxLen))
                sbox = [tL_x, tL_y, l_x, l_y]
                # Prepare label for this box
                bboxLabel = np.zeros(max(len(self.selected_attrs),1))
                # Test for overlap with foreground objects
                noOverlap = True
                #if len(cbboxList):
                for bb in cbboxList:
                    iou, aInb, bIna = computeIOU(sbox, bb['bbox'])
                    if iou > maxIou or aInb >0.8:
                        noOverlap = False
                    if bIna > 0.8:
                        bboxLabel[self.sattr_to_idx[self.catid2attr[bb['cid']]]] = 1
                if noOverlap and ((max_area < 0) or ((sbox[2]*sbox[3])< max_area) or (n_t>5)):
                    return sbox, bboxLabel, cbid
            n_t += 1

    def __getitem__(self, index):
        # In this situation ignore index and sample classes uniformly
        if self.balance_classes:
            currCls = random.choice(self.attToImgId.keys())
            index = random.choice(self.attToImgId[currCls])
        else:
            currCls = random.choice(self.catsInImg[index])

        cid = [self.sattr_to_idx[currCls]] if currCls != 'bg' else [0]

        returnvals = self.getbyIndexAndclass(index, cid)

        return tuple(returnvals)

    def getbyIdAndclass(self, imgid, cls, hflip=0):
        index = self.imgId2idx[imgid]
        cid = [self.sattr_to_idx[cls]] if cls != 'bg' else [0]
        returnvals = self.getbyIndexAndclass(index, cid)
        return tuple(returnvals)

    def getbyIndexAndclass(self, index, cid):

        image = Image.open(os.path.join(self.image_path, self.dataset['images'][index]['filename']))
        currCls = self.selected_attrs[cid[0]]
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')

        sampbbox, bboxLabel, cbid = self.randomBBoxSample(index, 0.5)
        extra_boxes = []
        if self.n_boxes > 1:
            # Sample random number of boxes between 1 and n_boxes
            c_nbox = np.random.randint(0,self.n_boxes)
            c_area = sampbbox[2]*sampbbox[3]
            for i in xrange(c_nbox):
                # Also stop at total area > 50%
                if c_area < 0.5:
                    bsamp, _, _ = self.randomBBoxSample(index, 0.6-c_area) # Extra 10% to make the sampling easier
                    extra_boxes.append(bsamp)
                    c_area += bsamp[2]*bsamp[3]
                else:
                    break

        label = self.dataset['images'][index]['label']

        # Apply transforms to the image.
        image = self.transform[0](image)
        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            sampbbox[0] = 1.0-(sampbbox[0]+sampbbox[2])
        if self.use_gt_mask==2:
            # Use GT boxes as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== currCls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                gtMask[0,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1.
        elif self.use_gt_mask==3:
            # Use GT centerpoints as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== currCls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                cent = [bbox[0] + bbox[2]//2, bbox[1]+bbox[3]//2]
                # center is marked by a 3x3 square patch
                gtMask[0,cent[1]-1:cent[1]+2,cent[0]-1:cent[0]+2] = 1.


        #Convert BBox to actual co-ordinates
        sampbbox = [int(bc*self.out_img_size) for bc in sampbbox]
        boxCrop = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
        # Create Mask
        mask = torch.zeros(1,self.out_img_size,self.out_img_size)
        mask[0,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
        if self.n_boxes > 1 and len(extra_boxes):
            for box in extra_boxes:
                box = [int(bc*self.out_img_size) for bc in box]
                mask[0,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1.

        if self.boxrotate:
            mask = torch.FloatTensor(np.asarray(self.rotateTrans(Image.fromarray(mask.numpy()[0]))))[None,::]
        if self.use_gt_mask:
            mask = torch.cat([mask, gtMask], dim=0)

        return self.transform[-1](image), torch.FloatTensor(label), self.transform[-1](boxCrop), torch.FloatTensor(bboxLabel), mask, torch.IntTensor(sampbbox), torch.LongTensor(cid)

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        return self.dataset['images'][index]['filename']

    def getcocoid(self, index):
        return self.dataset['images'][index]['id']

    def getGTMaskInp(self, index, cls, hflip=False, mask_type=None):
        what_mask = self.use_gt_mask if mask_type is None else mask_type
        if what_mask==1:
            print 'not supported'
            assert(0)
        elif what_mask==2:
            # Use GT boxes as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== cls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                gtMask[0,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1.
        elif what_mask==3:
            # Use GT centerpoints as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== cls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                cent = [bbox[0] + bbox[2]//2, bbox[1]+bbox[3]//2]
                # center is marked by a 3x3 square patch
                gtMask[0,cent[1]-1:cent[1]+2,cent[0]-1:cent[0]+2] = 1.
        else:
            gtMask = None

        return gtMask

class UnrelBBoxSample(Dataset):
    def __init__(self, transform, mode, select_attrs=[], datafile='dataset.json', out_img_size=128, bbox_out_size=64,
                 balance_classes=0, onlyrandBoxes=False, max_object_size=0., max_with_union=True, use_gt_mask=False,
                 boxrotate=0, n_boxes = 1):
        COCO_classes = ['person' , 'bicycle' , 'car' , 'motorcycle' , 'airplane' , 'bus' , 'train' , 'truck' , 'boat' , 'traffic light' , 'fire hydrant' , 'stop sign' , 'parking meter' , 'bench' , 'bird' , 'cat' , 'dog' , 'horse' , 'sheep' , 'cow' , 'elephant' , 'bear' , 'zebra' , 'giraffe' , 'backpack' , 'umbrella' , 'handbag' , 'tie' , 'suitcase' , 'frisbee' , 'skis' , 'snowboard' , 'sports ball' , 'kite' , 'baseball bat' , 'baseball glove' , 'skateboard' , 'surfboard' , 'tennis racket' , 'bottle' , 'wine glass' , 'cup' , 'fork' , 'knife' , 'spoon' , 'bowl' , 'banana' , 'apple' , 'sandwich' , 'orange' , 'broccoli' , 'carrot' , 'hot dog' , 'pizza' , 'donut' , 'cake' , 'chair' , 'couch' , 'potted plant' , 'bed' , 'dining table' , 'toilet' , 'tv' , 'laptop' , 'mouse' , 'remote' , 'keyboard' , 'cell phone' , 'microwave' , 'oven' , 'toaster' , 'sink' , 'refrigerator' , 'book' , 'clock' , 'vase' , 'scissors' , 'teddy bear' , 'hair drier' , 'toothbrush']
        self.use_cococlass = 1
        self.image_path = os.path.join('data','unrel','images')
        self.transform = transform
        self.mode = mode
        self.n_boxes = n_boxes
        self.iouThresh = 0.5
        self.dataset = json.load(open(os.path.join('data','unrel',datafile),'r'))
        self.num_data = len(self.dataset['images'])
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.out_img_size = out_img_size

        self.bbox_out_size = bbox_out_size
        #self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.selected_attrs = COCO_classes if len(select_attrs) == 0 else select_attrs
        self.balance_classes = balance_classes
        self.onlyrandBoxes = onlyrandBoxes
        self.max_object_size = max_object_size
        self.max_with_union= max_with_union
        self.use_gt_mask = 0
        self.boxrotate = boxrotate
        if self.boxrotate:
            self.rotateTrans = transforms.Compose([transforms.RandomRotation(boxrotate,resample=Image.NEAREST)])
        #if use_gt_mask == 1:
        #    print ' Not Supported'
        #    assert(0)

        self.randHFlip = 'Flip' in transform

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')
        self.imgId2idx = {imid:i for i,imid in enumerate(self.valid_ids)}

        self.num_data = len(self.dataset['images'])

    def preprocess(self):
        for i, attr in enumerate(self.dataset['categories']):
            self.attr2idx[attr['name']] = i
            self.idx2attr[i] = attr['name']
            self.catid2attr[attr['id']] = attr['name']

        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}

        # First remove unwanted splits:
        self.dataset['images'] = [img for img in self.dataset['images'] if img['split'] == self.mode]
        if self.max_object_size > 0.:
            validImgs = []
            for img in self.dataset['images']:
                if not self.max_with_union:
                    maxSize = max([bb['bbox'][2]*bb['bbox'][3] for bb in img['bboxAnn']])
                else:
                    boxByCls = defaultdict(list)
                    for bb in img['bboxAnn']:
                        boxByCls[bb['cid']].append(bb['bbox'])
                    unionAreas = [computeUnionArea(boxes) for cid,boxes in boxByCls.iteritems()]
                    maxSize = max(unionAreas)
                if maxSize < self.max_object_size:
                    validImgs.append(img)
            print ' %d of %d images left after size filtering'%(len(validImgs), len(self.dataset['images']))
            self.dataset['images'] = validImgs

        self.valid_ids = [img['id'] for img in self.dataset['images']]
        self.catsInImg = {}

        selset = set(self.selected_attrs)
        for i, img in enumerate(self.dataset['images']):
            self.dataset['images'][i]['label'] = np.zeros(max(len(selset),1))
            self.dataset['images'][i]['bboxAnn'] = [bb for bb in img['bboxAnn'] if bb['cococlass'] in selset]

            # Correct BBox for Resize(of smaller edge) and CenterCrop
            fixedbbox = []
            imgSize = self.dataset['images'][i]['imgSize']
            maxSide = np.argmax(imgSize)
            for j in xrange(len(self.dataset['images'][i]['bboxAnn'])):
                cbbox = self.dataset['images'][i]['bboxAnn'][j]
                maxSideLen = int(float(self.out_img_size * imgSize[maxSide]) / (imgSize[1-maxSide]))
                assert(maxSideLen >= self.out_img_size)
                newStartCord = round((maxSideLen - self.out_img_size)/2.)
                boxStart = min( max(cbbox['bbox'][maxSide]*maxSideLen - newStartCord, 0),  self.out_img_size)
                boxEnd =  min(max((cbbox['bbox'][maxSide]+cbbox['bbox'][maxSide+2])*maxSideLen - newStartCord, 0), self.out_img_size)
                length = boxEnd - boxStart
                if length > 5:
                    cbbox['bbox'][maxSide] = float(boxStart)/self.out_img_size
                    cbbox['bbox'][maxSide+2] = float(length)/self.out_img_size
                    if cbbox['bbox'][1-maxSide+2] >= 0.04:
                        fixedbbox.append(cbbox)
                        if cbbox['bbox'][0]<0. or cbbox['bbox'][1] < 0. or cbbox['bbox'][0]>1.0 or cbbox['bbox'][1]> 1.0:
                            import ipdb; ipdb.set_trace()
            self.dataset['images'][i]['bboxAnn'] = fixedbbox
            self.dataset['images'][i]['label'][[self.sattr_to_idx[bb['cococlass']] for bb in img['bboxAnn']]] = 1.

            # Convert bbox data to numpy arrays
            #for j, bb in enumerate(self.dataset['images'][i]['bboxAnn']):
            #    self.dataset['images'][i]['bboxAnn'][j]['bbox'] = np.array(bb['bbox'])
            # Create bbox labels.
            for j, bb in enumerate(self.dataset['images'][i]['bboxAnn']):
                #Check for IOU > 0.5 with other bbox
                iouAr = [computeContainment(bb['bbox'], bother['bbox'])[0] for bother in self.dataset['images'][i]['bboxAnn']]
                self.dataset['images'][i]['bboxAnn'][j]['box_label'] = np.zeros(len(selset))
                self.dataset['images'][i]['bboxAnn'][j]['box_label'][[self.sattr_to_idx[self.dataset['images'][i]['bboxAnn'][ii]['cococlass']] for ii,iv in enumerate(iouAr) if iv>self.iouThresh]] = 1.

        self.attToImgId = defaultdict(set)
        for i, img in enumerate(self.dataset['images']):
            classesInImg = [bb['cococlass'] for bb in img['bboxAnn'] if bb['cococlass'] in selset]
            if len(classesInImg):
                self.catsInImg[i] = classesInImg
                for att in classesInImg:
                    self.attToImgId[att].add(i)
            else:
                self.attToImgId['bg'].add(i)
                self.catsInImg[i] = ['bg']
        self.attToImgId = {k:list(v) for k,v in self.attToImgId.iteritems()}


    def randomBBoxSample(self, index, max_area = -1):
        # With 50% chance sample from background or foreground
        # Minimum size
        minLen = 0.1
        maxLen = 0.85
        maxIou = 0.3
        cbboxList = self.dataset['images'][index]['bboxAnn'] if not self.onlyrandBoxes else []
        n_t = 0
        while 1:
            if len(cbboxList) and (random.random()<0.9):
                cbid = random.randrange(len(cbboxList))
                sbox = self.dataset['images'][index]['bboxAnn'][cbid]
                return copy(sbox['bbox']),sbox['box_label'], cbid
            else:
                # sample a random background box
                cbid = None
                tL_x, tL_y = random.uniform(0,1.-minLen-0.01), random.uniform(0,1.-minLen-0.01)
                l_x = random.uniform(minLen, min(1.-tL_x,maxLen))
                l_y = random.uniform(minLen, min(1.-tL_y,maxLen))
                sbox = [tL_x, tL_y, l_x, l_y]
                # Prepare label for this box
                bboxLabel = np.zeros(max(len(self.selected_attrs),1))
                # Test for overlap with foreground objects
                noOverlap = True
                #if len(cbboxList):
                for bb in cbboxList:
                    iou, aInb, bIna = computeIOU(sbox, bb['bbox'])
                    if iou > maxIou or aInb >0.8:
                        noOverlap = False
                    if bIna > 0.8:
                        bboxLabel[self.sattr_to_idx[bb['cococlass']]] = 1
                if noOverlap and ((max_area < 0) or ((sbox[2]*sbox[3])< max_area) or (n_t>5)):
                    return sbox, bboxLabel, cbid
            n_t += 1

    def __getitem__(self, index):
        # In this situation ignore index and sample classes uniformly
        if self.balance_classes:
            currCls = random.choice(self.attToImgId.keys())
            index = random.choice(self.attToImgId[currCls])
        else:
            currCls = random.choice(self.catsInImg[index])

        cid = [self.sattr_to_idx[currCls]] if currCls != 'bg' else [0]

        returnvals = self.getbyIndexAndclass(index, cid)

        return tuple(returnvals)

    def getbyIdAndclass(self, imgid, cls, hflip=0):
        index = self.imgId2idx[imgid]
        cid = [self.sattr_to_idx[cls]] if cls != 'bg' else [0]
        returnvals = self.getbyIndexAndclass(index, cid)
        return tuple(returnvals)

    def getbyIndexAndclass(self, index, cid):

        image = Image.open(os.path.join(self.image_path, self.dataset['images'][index]['filename']))
        currCls = self.selected_attrs[cid[0]]
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')

        sampbbox, bboxLabel, cbid = self.randomBBoxSample(index, 0.5)
        extra_boxes = []
        if self.n_boxes > 1:
            # Sample random number of boxes between 1 and n_boxes
            c_nbox = np.random.randint(0,self.n_boxes)
            c_area = sampbbox[2]*sampbbox[3]
            for i in xrange(c_nbox):
                # Also stop at total area > 50%
                if c_area < 0.7:
                    bsamp, _, _ = self.randomBBoxSample(index, 0.8-c_area) # Extra 10% to make the sampling easier
                    extra_boxes.append(bsamp)
                    c_area += bsamp[2]*bsamp[3]
                else:
                    break

        label = self.dataset['images'][index]['label']

        # Apply transforms to the image.
        image = self.transform[0](image)
        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            sampbbox[0] = 1.0-(sampbbox[0]+sampbbox[2])
        if self.use_gt_mask==2:
            # Use GT boxes as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== currCls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                gtMask[0,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1.
        elif self.use_gt_mask==3:
            # Use GT centerpoints as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== currCls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                cent = [bbox[0] + bbox[2]//2, bbox[1]+bbox[3]//2]
                # center is marked by a 3x3 square patch
                gtMask[0,cent[1]-1:cent[1]+2,cent[0]-1:cent[0]+2] = 1.


        #Convert BBox to actual co-ordinates
        sampbbox = [int(bc*self.out_img_size) for bc in sampbbox]
        boxCrop = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
        # Create Mask
        mask = torch.zeros(1,self.out_img_size,self.out_img_size)
        mask[0,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
        if self.n_boxes > 1 and len(extra_boxes):
            for box in extra_boxes:
                box = [int(bc*self.out_img_size) for bc in box]
                mask[0,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1.

        if self.boxrotate:
            mask = torch.FloatTensor(np.asarray(self.rotateTrans(Image.fromarray(mask.numpy()[0]))))[None,::]
        if self.use_gt_mask:
            mask = torch.cat([mask, gtMask], dim=0)

        return self.transform[-1](image), torch.FloatTensor(label), self.transform[-1](boxCrop), torch.FloatTensor(bboxLabel), mask, torch.IntTensor(sampbbox), torch.LongTensor(cid)

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        return self.dataset['images'][index]['filename']

    def getcocoid(self, index):
        return self.dataset['images'][index]['id']

    def getGTMaskInp(self, index, cls, hflip=False, mask_type=None):
        what_mask = self.use_gt_mask if mask_type is None else mask_type
        if what_mask==1:
            print 'not supported'
            assert(0)
        elif what_mask==2:
            # Use GT boxes as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== cls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                gtMask[0,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1.
        elif what_mask==3:
            # Use GT centerpoints as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== cls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                cent = [bbox[0] + bbox[2]//2, bbox[1]+bbox[3]//2]
                # center is marked by a 3x3 square patch
                gtMask[0,cent[1]-1:cent[1]+2,cent[0]-1:cent[0]+2] = 1.
        else:
            gtMask = None

        return gtMask


class OutofContextBBoxSample(Dataset):
    def __init__(self, transform, mode, select_attrs=[], datafile='dataset.json', out_img_size=128, bbox_out_size=64,
                 balance_classes=0, onlyrandBoxes=False, max_object_size=0., max_with_union=True, use_gt_mask=False,
                 boxrotate=0, n_boxes = 1):
        COCO_classes = ['person' , 'bicycle' , 'car' , 'motorcycle' , 'airplane' , 'bus' , 'train' , 'truck' , 'boat' , 'traffic light' , 'fire hydrant' , 'stop sign' , 'parking meter' , 'bench' , 'bird' , 'cat' , 'dog' , 'horse' , 'sheep' , 'cow' , 'elephant' , 'bear' , 'zebra' , 'giraffe' , 'backpack' , 'umbrella' , 'handbag' , 'tie' , 'suitcase' , 'frisbee' , 'skis' , 'snowboard' , 'sports ball' , 'kite' , 'baseball bat' , 'baseball glove' , 'skateboard' , 'surfboard' , 'tennis racket' , 'bottle' , 'wine glass' , 'cup' , 'fork' , 'knife' , 'spoon' , 'bowl' , 'banana' , 'apple' , 'sandwich' , 'orange' , 'broccoli' , 'carrot' , 'hot dog' , 'pizza' , 'donut' , 'cake' , 'chair' , 'couch' , 'potted plant' , 'bed' , 'dining table' , 'toilet' , 'tv' , 'laptop' , 'mouse' , 'remote' , 'keyboard' , 'cell phone' , 'microwave' , 'oven' , 'toaster' , 'sink' , 'refrigerator' , 'book' , 'clock' , 'vase' , 'scissors' , 'teddy bear' , 'hair drier' , 'toothbrush']
        self.use_cococlass = 1
        self.image_path = os.path.join('data','outofcontext','images')
        self.transform = transform
        self.mode = mode
        self.n_boxes = n_boxes
        self.iouThresh = 0.5
        self.dataset = json.load(open(os.path.join('data','outofcontext',datafile),'r'))
        self.num_data = len(self.dataset['images'])
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.out_img_size = out_img_size

        self.bbox_out_size = bbox_out_size
        #self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.selected_attrs = COCO_classes if len(select_attrs) == 0 else select_attrs
        self.balance_classes = balance_classes
        self.onlyrandBoxes = onlyrandBoxes
        self.max_object_size = max_object_size
        self.max_with_union= max_with_union
        self.use_gt_mask = 0
        self.boxrotate = boxrotate
        if self.boxrotate:
            self.rotateTrans = transforms.Compose([transforms.RandomRotation(boxrotate,resample=Image.NEAREST)])
        #if use_gt_mask == 1:
        #    print ' Not Supported'
        #    assert(0)

        self.randHFlip = 'Flip' in transform

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')
        self.imgId2idx = {imid:i for i,imid in enumerate(self.valid_ids)}

        self.num_data = len(self.dataset['images'])

    def preprocess(self):
        for i, attr in enumerate(self.dataset['categories']):
            self.attr2idx[attr['name']] = i
            self.idx2attr[i] = attr['name']
            self.catid2attr[attr['id']] = attr['name']

        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}

        # First remove unwanted splits:
        self.dataset['images'] = [img for img in self.dataset['images'] if img['split'] == self.mode]
        if self.max_object_size > 0.:
            validImgs = []
            for img in self.dataset['images']:
                if not self.max_with_union:
                    maxSize = max([bb['bbox'][2]*bb['bbox'][3] for bb in img['bboxAnn']])
                else:
                    boxByCls = defaultdict(list)
                    for bb in img['bboxAnn']:
                        boxByCls[bb['cid']].append(bb['bbox'])
                    unionAreas = [computeUnionArea(boxes) for cid,boxes in boxByCls.iteritems()]
                    maxSize = max(unionAreas)
                if maxSize < self.max_object_size:
                    validImgs.append(img)
            print ' %d of %d images left after size filtering'%(len(validImgs), len(self.dataset['images']))
            self.dataset['images'] = validImgs

        self.valid_ids = [img['id'] for img in self.dataset['images']]
        self.catsInImg = {}

        selset = set(self.selected_attrs)
        for i, img in enumerate(self.dataset['images']):
            self.dataset['images'][i]['label'] = np.zeros(max(len(selset),1))
            self.dataset['images'][i]['bboxAnn'] = [bb for bb in img['bboxAnn'] if bb['cococlass'] in selset]# and bb['outofcontext'] == 1]

            # Correct BBox for Resize(of smaller edge) and CenterCrop
            fixedbbox = []
            imgSize = self.dataset['images'][i]['imgSize']
            maxSide = np.argmax(imgSize)
            for j in xrange(len(self.dataset['images'][i]['bboxAnn'])):
                cbbox = self.dataset['images'][i]['bboxAnn'][j]
                maxSideLen = int(float(self.out_img_size * imgSize[maxSide]) / (imgSize[1-maxSide]))
                assert(maxSideLen >= self.out_img_size)
                newStartCord = round((maxSideLen - self.out_img_size)/2.)
                boxStart = min( max(cbbox['bbox'][maxSide]*maxSideLen - newStartCord, 0),  self.out_img_size)
                boxEnd =  min(max((cbbox['bbox'][maxSide]+cbbox['bbox'][maxSide+2])*maxSideLen - newStartCord, 0), self.out_img_size)
                length = boxEnd - boxStart
                if length > 5:
                    cbbox['bbox'][maxSide] = float(boxStart)/self.out_img_size
                    cbbox['bbox'][maxSide+2] = float(length)/self.out_img_size
                    if cbbox['bbox'][1-maxSide+2] >= 0.04:
                        fixedbbox.append(cbbox)
                        if cbbox['bbox'][0]<0. or cbbox['bbox'][1] < 0. or cbbox['bbox'][0]>1.0 or cbbox['bbox'][1]> 1.0:
                            import ipdb; ipdb.set_trace()
            self.dataset['images'][i]['bboxAnn'] = fixedbbox
            self.dataset['images'][i]['label'][[self.sattr_to_idx[bb['cococlass']] for bb in img['bboxAnn']]] = 1.

            # Convert bbox data to numpy arrays
            #for j, bb in enumerate(self.dataset['images'][i]['bboxAnn']):
            #    self.dataset['images'][i]['bboxAnn'][j]['bbox'] = np.array(bb['bbox'])
            # Create bbox labels.
            for j, bb in enumerate(self.dataset['images'][i]['bboxAnn']):
                #Check for IOU > 0.5 with other bbox
                iouAr = [computeContainment(bb['bbox'], bother['bbox'])[0] for bother in self.dataset['images'][i]['bboxAnn']]
                self.dataset['images'][i]['bboxAnn'][j]['box_label'] = np.zeros(len(selset))
                self.dataset['images'][i]['bboxAnn'][j]['box_label'][[self.sattr_to_idx[self.dataset['images'][i]['bboxAnn'][ii]['cococlass']] for ii,iv in enumerate(iouAr) if iv>self.iouThresh]] = 1.

        self.attToImgId = defaultdict(set)
        for i, img in enumerate(self.dataset['images']):
            classesInImg = [bb['cococlass'] for bb in img['bboxAnn'] if bb['cococlass'] in selset]
            if len(classesInImg):
                self.catsInImg[i] = classesInImg
                for att in classesInImg:
                    self.attToImgId[att].add(i)
            else:
                self.attToImgId['bg'].add(i)
                self.catsInImg[i] = ['bg']
        self.attToImgId = {k:list(v) for k,v in self.attToImgId.iteritems()}


    def randomBBoxSample(self, index, max_area = -1):
        # With 50% chance sample from background or foreground
        # Minimum size
        minLen = 0.1
        maxLen = 0.85
        maxIou = 0.3
        cbboxList = self.dataset['images'][index]['bboxAnn'] if not self.onlyrandBoxes else []
        n_t = 0
        while 1:
            if len(cbboxList) and (random.random()<0.9):
                cbid = random.randrange(len(cbboxList))
                sbox = self.dataset['images'][index]['bboxAnn'][cbid]
                return copy(sbox['bbox']),sbox['box_label'], cbid
            else:
                # sample a random background box
                cbid = None
                tL_x, tL_y = random.uniform(0,1.-minLen-0.01), random.uniform(0,1.-minLen-0.01)
                l_x = random.uniform(minLen, min(1.-tL_x,maxLen))
                l_y = random.uniform(minLen, min(1.-tL_y,maxLen))
                sbox = [tL_x, tL_y, l_x, l_y]
                # Prepare label for this box
                bboxLabel = np.zeros(max(len(self.selected_attrs),1))
                # Test for overlap with foreground objects
                noOverlap = True
                #if len(cbboxList):
                for bb in cbboxList:
                    iou, aInb, bIna = computeIOU(sbox, bb['bbox'])
                    if iou > maxIou or aInb >0.8:
                        noOverlap = False
                    if bIna > 0.8:
                        bboxLabel[self.sattr_to_idx[bb['cococlass']]] = 1
                if noOverlap and ((max_area < 0) or ((sbox[2]*sbox[3])< max_area) or (n_t>5)):
                    return sbox, bboxLabel, cbid
            n_t += 1

    def __getitem__(self, index):
        # In this situation ignore index and sample classes uniformly
        if self.balance_classes:
            currCls = random.choice(self.attToImgId.keys())
            index = random.choice(self.attToImgId[currCls])
        else:
            currCls = random.choice(self.catsInImg[index])

        cid = [self.sattr_to_idx[currCls]] if currCls != 'bg' else [0]

        returnvals = self.getbyIndexAndclass(index, cid)

        return tuple(returnvals)

    def getbyIdAndclass(self, imgid, cls, hflip=0):
        index = self.imgId2idx[imgid]
        cid = [self.sattr_to_idx[cls]] if cls != 'bg' else [0]
        returnvals = self.getbyIndexAndclass(index, cid)
        return tuple(returnvals)

    def getbyIndexAndclass(self, index, cid):

        image = Image.open(os.path.join(self.image_path, self.dataset['images'][index]['filename']))
        currCls = self.selected_attrs[cid[0]]
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')

        sampbbox, bboxLabel, cbid = self.randomBBoxSample(index, 0.5)
        extra_boxes = []
        if self.n_boxes > 1:
            # Sample random number of boxes between 1 and n_boxes
            c_nbox = np.random.randint(0,self.n_boxes)
            c_area = sampbbox[2]*sampbbox[3]
            for i in xrange(c_nbox):
                # Also stop at total area > 50%
                if c_area < 0.7:
                    bsamp, _, _ = self.randomBBoxSample(index, 0.8-c_area) # Extra 10% to make the sampling easier
                    extra_boxes.append(bsamp)
                    c_area += bsamp[2]*bsamp[3]
                else:
                    break

        label = self.dataset['images'][index]['label']

        # Apply transforms to the image.
        image = self.transform[0](image)
        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            sampbbox[0] = 1.0-(sampbbox[0]+sampbbox[2])
        if self.use_gt_mask==2:
            # Use GT boxes as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== currCls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                gtMask[0,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1.
        elif self.use_gt_mask==3:
            # Use GT centerpoints as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== currCls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                cent = [bbox[0] + bbox[2]//2, bbox[1]+bbox[3]//2]
                # center is marked by a 3x3 square patch
                gtMask[0,cent[1]-1:cent[1]+2,cent[0]-1:cent[0]+2] = 1.


        #Convert BBox to actual co-ordinates
        sampbbox = [int(bc*self.out_img_size) for bc in sampbbox]
        boxCrop = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
        # Create Mask
        mask = torch.zeros(1,self.out_img_size,self.out_img_size)
        mask[0,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
        if self.n_boxes > 1 and len(extra_boxes):
            for box in extra_boxes:
                box = [int(bc*self.out_img_size) for bc in box]
                mask[0,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1.

        if self.boxrotate:
            mask = torch.FloatTensor(np.asarray(self.rotateTrans(Image.fromarray(mask.numpy()[0]))))[None,::]
        if self.use_gt_mask:
            mask = torch.cat([mask, gtMask], dim=0)

        return self.transform[-1](image), torch.FloatTensor(label), self.transform[-1](boxCrop), torch.FloatTensor(bboxLabel), mask, torch.IntTensor(sampbbox), torch.LongTensor(cid)

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        return self.dataset['images'][index]['filename']

    def getcocoid(self, index):
        return self.dataset['images'][index]['id']

    def getGTMaskInp(self, index, cls, hflip=False, mask_type=None):
        what_mask = self.use_gt_mask if mask_type is None else mask_type
        if what_mask==1:
            print 'not supported'
            assert(0)
        elif what_mask==2:
            # Use GT boxes as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== cls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                gtMask[0,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1.
        elif what_mask==3:
            # Use GT centerpoints as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== cls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                cent = [bbox[0] + bbox[2]//2, bbox[1]+bbox[3]//2]
                # center is marked by a 3x3 square patch
                gtMask[0,cent[1]-1:cent[1]+2,cent[0]-1:cent[0]+2] = 1.
        else:
            gtMask = None

        return gtMask



class FlickrLogoBBoxSample(Dataset):
    def __init__(self, transform, mode, select_attrs=[], datafile='dataset.json', out_img_size=128, bbox_out_size=64,
                 balance_classes=0, onlyrandBoxes=False, max_object_size=0., max_with_union=True, use_gt_mask=False,
                 boxrotate=0, n_boxes = 1):
        self.image_path = os.path.join('data','flickr_logos_27_dataset','flickr_logos_27_dataset_images')
        self.transform = transform
        self.mode = mode
        self.n_boxes = n_boxes
        self.iouThresh = 0.5
        self.dataset = json.load(open(os.path.join('data','flickr_logos_27_dataset',datafile),'r'))
        self.num_data = len(self.dataset['images'])
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.out_img_size = out_img_size
        self.bbox_out_size = bbox_out_size
        #self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.selected_attrs = select_attrs
        self.balance_classes = balance_classes
        self.onlyrandBoxes = onlyrandBoxes
        self.max_object_size = max_object_size
        self.max_with_union= max_with_union
        self.use_gt_mask = use_gt_mask
        self.boxrotate = boxrotate
        if self.boxrotate:
            self.rotateTrans = transforms.Compose([transforms.RandomRotation(boxrotate,resample=Image.NEAREST)])
        if use_gt_mask == 1:
            print ' Not Supported'
            assert(0)

        self.randHFlip = 'Flip' in transform

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')
        self.imgId2idx = {imid:i for i,imid in enumerate(self.valid_ids)}

        self.num_data = len(self.dataset['images'])

    def preprocess(self):
        for i, attr in enumerate(self.dataset['categories']):
            self.attr2idx[attr['name']] = i
            self.idx2attr[i] = attr['name']
            self.catid2attr[attr['id']] = attr['name']

        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}

        # First remove unwanted splits:
        self.dataset['images'] = [img for img in self.dataset['images'] if img['split'] == self.mode]
        if self.max_object_size > 0.:
            validImgs = []
            for img in self.dataset['images']:
                if not self.max_with_union:
                    maxSize = max([bb['bbox'][2]*bb['bbox'][3] for bb in img['bboxAnn']])
                else:
                    boxByCls = defaultdict(list)
                    for bb in img['bboxAnn']:
                        boxByCls[bb['cid']].append(bb['bbox'])
                    unionAreas = [computeUnionArea(boxes) for cid,boxes in boxByCls.iteritems()]
                    maxSize = max(unionAreas)
                if maxSize < self.max_object_size:
                    validImgs.append(img)
            print ' %d of %d images left after size filtering'%(len(validImgs), len(self.dataset['images']))
            self.dataset['images'] = validImgs

        self.valid_ids = [img['id'] for img in self.dataset['images']]
        self.catsInImg = {}

        selset = set(self.selected_attrs)
        for i, img in enumerate(self.dataset['images']):
            self.dataset['images'][i]['label'] = np.zeros(max(len(selset),1))
            self.dataset['images'][i]['bboxAnn'] = [bb for bb in img['bboxAnn'] if self.catid2attr[bb['cid']] in selset]

            # Correct BBox for Resize(of smaller edge) and CenterCrop
            fixedbbox = []
            imgSize = self.dataset['images'][i]['imgSize']
            maxSide = np.argmax(imgSize)
            for j in xrange(len(self.dataset['images'][i]['bboxAnn'])):
                cbbox = self.dataset['images'][i]['bboxAnn'][j]
                maxSideLen = int(float(self.out_img_size * imgSize[maxSide]) / (imgSize[1-maxSide]))
                assert(maxSideLen >= self.out_img_size)
                newStartCord = round((maxSideLen - self.out_img_size)/2.)
                boxStart = min( max(cbbox['bbox'][maxSide]*maxSideLen - newStartCord, 0),  self.out_img_size)
                boxEnd =  min(max((cbbox['bbox'][maxSide]+cbbox['bbox'][maxSide+2])*maxSideLen - newStartCord, 0), self.out_img_size)
                length = boxEnd - boxStart
                if length > 5:
                    cbbox['bbox'][maxSide] = float(boxStart)/self.out_img_size
                    cbbox['bbox'][maxSide+2] = float(length)/self.out_img_size
                    if cbbox['bbox'][1-maxSide+2] >= 0.04:
                        fixedbbox.append(cbbox)
                        if cbbox['bbox'][0]<0. or cbbox['bbox'][1] < 0. or cbbox['bbox'][0]>1.0 or cbbox['bbox'][1]> 1.0:
                            import ipdb; ipdb.set_trace()
            self.dataset['images'][i]['bboxAnn'] = fixedbbox
            self.dataset['images'][i]['label'][[self.sattr_to_idx[self.catid2attr[bb['cid']]] for bb in img['bboxAnn']]] = 1.

            # Convert bbox data to numpy arrays
            #for j, bb in enumerate(self.dataset['images'][i]['bboxAnn']):
            #    self.dataset['images'][i]['bboxAnn'][j]['bbox'] = np.array(bb['bbox'])
            # Create bbox labels.
            for j, bb in enumerate(self.dataset['images'][i]['bboxAnn']):
                #Check for IOU > 0.5 with other bbox
                iouAr = [computeContainment(bb['bbox'], bother['bbox'])[0] for bother in self.dataset['images'][i]['bboxAnn']]
                self.dataset['images'][i]['bboxAnn'][j]['box_label'] = np.zeros(len(selset))
                self.dataset['images'][i]['bboxAnn'][j]['box_label'][[self.sattr_to_idx[self.catid2attr[self.dataset['images'][i]['bboxAnn'][ii]['cid']]] for ii,iv in enumerate(iouAr) if iv>self.iouThresh]] = 1.

        self.attToImgId = defaultdict(set)
        for i, img in enumerate(self.dataset['images']):
            classesInImg = [self.catid2attr[bb['cid']] for bb in img['bboxAnn'] if self.catid2attr[bb['cid']] in selset]
            if len(classesInImg):
                self.catsInImg[i] = classesInImg
                for att in classesInImg:
                    self.attToImgId[att].add(i)
            else:
                self.attToImgId['bg'].add(i)
                self.catsInImg[i] = ['bg']
        self.attToImgId = {k:list(v) for k,v in self.attToImgId.iteritems()}


    def randomBBoxSample(self, index, max_area = -1):
        # With 50% chance sample from background or foreground
        # Minimum size
        minLen = 0.1
        maxLen = 0.85
        maxIou = 0.3
        cbboxList = self.dataset['images'][index]['bboxAnn'] if not self.onlyrandBoxes else []
        n_t = 0
        while 1:
            if len(cbboxList) and (random.random()<0.9):
                cbid = random.randrange(len(cbboxList))
                sbox = self.dataset['images'][index]['bboxAnn'][cbid]
                return copy(sbox['bbox']),sbox['box_label'], cbid
            else:
                # sample a random background box
                cbid = None
                tL_x, tL_y = random.uniform(0,1.-minLen-0.01), random.uniform(0,1.-minLen-0.01)
                l_x = random.uniform(minLen, min(1.-tL_x,maxLen))
                l_y = random.uniform(minLen, min(1.-tL_y,maxLen))
                sbox = [tL_x, tL_y, l_x, l_y]
                # Prepare label for this box
                bboxLabel = np.zeros(max(len(self.selected_attrs),1))
                # Test for overlap with foreground objects
                noOverlap = True
                #if len(cbboxList):
                for bb in cbboxList:
                    iou, aInb, bIna = computeIOU(sbox, bb['bbox'])
                    if iou > maxIou or aInb >0.8:
                        noOverlap = False
                    if bIna > 0.8:
                        bboxLabel[self.sattr_to_idx[self.catid2attr[bb['cid']]]] = 1
                if noOverlap and ((max_area < 0) or ((sbox[2]*sbox[3])< max_area) or (n_t>5)):
                    return sbox, bboxLabel, cbid
            n_t += 1

    def __getitem__(self, index):
        # In this situation ignore index and sample classes uniformly
        if self.balance_classes:
            currCls = random.choice(self.attToImgId.keys())
            index = random.choice(self.attToImgId[currCls])
        else:
            currCls = random.choice(self.catsInImg[index])

        cid = [self.sattr_to_idx[currCls]] if currCls != 'bg' else [0]

        returnvals = self.getbyIndexAndclass(index, cid)

        return tuple(returnvals)

    def getbyIdAndclass(self, imgid, cls, hflip=0):
        index = self.imgId2idx[imgid]
        cid = [self.sattr_to_idx[cls]] if cls != 'bg' else [0]
        returnvals = self.getbyIndexAndclass(index, cid)
        return tuple(returnvals)

    def getbyIndexAndclass(self, index, cid):

        image = Image.open(os.path.join(self.image_path, self.dataset['images'][index]['filename']))
        currCls = self.selected_attrs[cid[0]]
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')

        sampbbox, bboxLabel, cbid = self.randomBBoxSample(index, 0.5)
        extra_boxes = []
        if self.n_boxes > 1:
            # Sample random number of boxes between 1 and n_boxes
            c_nbox = np.random.randint(0,self.n_boxes)
            c_area = sampbbox[2]*sampbbox[3]
            for i in xrange(c_nbox):
                # Also stop at total area > 50%
                if c_area < 0.7:
                    bsamp, _, _ = self.randomBBoxSample(index, 0.8-c_area) # Extra 10% to make the sampling easier
                    extra_boxes.append(bsamp)
                    c_area += bsamp[2]*bsamp[3]
                else:
                    break

        label = self.dataset['images'][index]['label']

        # Apply transforms to the image.
        image = self.transform[0](image)
        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            sampbbox[0] = 1.0-(sampbbox[0]+sampbbox[2])
        if self.use_gt_mask==2:
            # Use GT boxes as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== currCls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                gtMask[0,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1.
        elif self.use_gt_mask==3:
            # Use GT centerpoints as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== currCls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                cent = [bbox[0] + bbox[2]//2, bbox[1]+bbox[3]//2]
                # center is marked by a 3x3 square patch
                gtMask[0,cent[1]-1:cent[1]+2,cent[0]-1:cent[0]+2] = 1.


        #Convert BBox to actual co-ordinates
        sampbbox = [int(bc*self.out_img_size) for bc in sampbbox]
        boxCrop = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
        # Create Mask
        mask = torch.zeros(1,self.out_img_size,self.out_img_size)
        mask[0,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
        if self.n_boxes > 1 and len(extra_boxes):
            for box in extra_boxes:
                box = [int(bc*self.out_img_size) for bc in box]
                mask[0,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1.

        if self.boxrotate:
            mask = torch.FloatTensor(np.asarray(self.rotateTrans(Image.fromarray(mask.numpy()[0]))))[None,::]
        if self.use_gt_mask:
            mask = torch.cat([mask, gtMask], dim=0)

        return self.transform[-1](image), torch.FloatTensor(label), self.transform[-1](boxCrop), torch.FloatTensor(bboxLabel), mask, torch.IntTensor(sampbbox), torch.LongTensor(cid)

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        return self.dataset['images'][index]['filename']

    def getcocoid(self, index):
        return self.dataset['images'][index]['id']

    def getGTMaskInp(self, index, cls, hflip=False, mask_type=None):
        what_mask = self.use_gt_mask if mask_type is None else mask_type
        if what_mask==1:
            print 'not supported'
            assert(0)
        elif what_mask==2:
            # Use GT boxes as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== cls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                gtMask[0,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1.
        elif what_mask==3:
            # Use GT centerpoints as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== cls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                cent = [bbox[0] + bbox[2]//2, bbox[1]+bbox[3]//2]
                # center is marked by a 3x3 square patch
                gtMask[0,cent[1]-1:cent[1]+2,cent[0]-1:cent[0]+2] = 1.
        else:
            gtMask = None

        return gtMask


class Places2DatasetBBoxSample(Dataset):
    def __init__(self, transform, mode, select_attrs=[], datafile='datasetBoxAnn.json', out_img_size=128, bbox_out_size=64,
                 balance_classes=0, onlyrandBoxes=False, max_object_size=0., max_with_union=True, use_gt_mask=False,
                 boxrotate=0, n_boxes = 1):
        self.image_path = os.path.join('data','places2','images')
        self.transform = transform
        self.mode = mode
        self.n_boxes = n_boxes
        self.iouThresh = 0.5
        self.filenames = open(os.path.join('data','places2',mode+'_files.txt'),'r').read().splitlines()
        self.num_data = len(self.filenames)
        self.out_img_size = out_img_size
        self.bbox_out_size = bbox_out_size
        #self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.selected_attrs = ['background']
        self.onlyrandBoxes = onlyrandBoxes
        self.max_object_size = max_object_size
        self.boxrotate = boxrotate
        if self.boxrotate:
            self.rotateTrans = transforms.Compose([transforms.RandomRotation(boxrotate,resample=Image.NEAREST)])

        self.randHFlip = 'Flip' in transform

        print ('Start preprocessing dataset..!')
        print ('Finished preprocessing dataset..!')

        self.valid_ids  = [int(fname.split('_')[-1].split('.')[0][-8:]) for fname in self.filenames]

    def randomBBoxSample(self, max_area = -1):
        # With 50% chance sample from background or foreground
        # Minimum size
        minLen = 0.1
        maxLen = 0.7
        maxIou = 0.3
        cbboxList = []
        n_t = 0
        while 1:
            # sample a random background box
            cbid = None
            tL_x, tL_y = random.uniform(0,1.-minLen-0.01), random.uniform(0,1.-minLen-0.01)
            l_x = random.uniform(minLen, min(1.-tL_x,maxLen))
            l_y = random.uniform(minLen, min(1.-tL_y,maxLen))
            sbox = [tL_x, tL_y, l_x, l_y]
            # Prepare label for this box
            bboxLabel = np.zeros(max(len(self.selected_attrs),1))
            #if len(cbboxList):
            if ((max_area < 0) or ((sbox[2]*sbox[3])< max_area) or (n_t>5)):
                return sbox, bboxLabel, cbid
            n_t += 1

    def __getitem__(self, index):
        # In this situation ignore index and sample classes uniformly
        image = Image.open(os.path.join(self.image_path,self.filenames[index]))
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')

        cid = [0]
        sampbbox, bboxLabel, cbid = self.randomBBoxSample(0.5)
        extra_boxes = []
        if self.n_boxes > 1:
            # Sample random number of boxes between 1 and n_boxes
            c_nbox = np.random.randint(0,self.n_boxes)
            c_area = sampbbox[2]*sampbbox[3]
            for i in xrange(c_nbox):
                # Also stop at total area > 50%
                if c_area < 0.5:
                    bsamp, _, _ = self.randomBBoxSample(0.6-c_area) # Extra 10% to make the sampling easier
                    extra_boxes.append(bsamp)
                    c_area += bsamp[2]*bsamp[3]
                else:
                    break

        label = np.zeros(max(len(self.selected_attrs),1))
        # Apply transforms to the image.
        image = self.transform[0](image)
        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            sampbbox[0] = 1.0-(sampbbox[0]+sampbbox[2])

        #Convert BBox to actual co-ordinates
        sampbbox = [int(bc*self.out_img_size) for bc in sampbbox]
        #Now obtain the crop
        boxCrop = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
        # Create Mask
        mask = torch.zeros(1,self.out_img_size,self.out_img_size)
        mask[0,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
        if self.n_boxes > 1 and len(extra_boxes):
            for box in extra_boxes:
                box = [int(bc*self.out_img_size) for bc in box]
                mask[0,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1.

        if self.boxrotate:
            mask = torch.FloatTensor(np.asarray(self.rotateTrans(Image.fromarray(mask.numpy()[0]))))[None,::]

        return self.transform[-1](image), torch.FloatTensor(label), self.transform[-1](boxCrop), torch.FloatTensor(bboxLabel), mask, torch.IntTensor(sampbbox), torch.LongTensor(cid)

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        return self.filenames[index]

    def getcocoid(self, index):
        return self.valid_ids[index]

    def getGTMaskInp(self, index, cls, hflip=False, mask_type=None):
        gtMask = None
        return gtMask

class PascalDatasetBBoxSample(Dataset):
    def __init__(self, transform, mode, select_attrs=[], datafile='dataset.json', out_img_size=128, bbox_out_size=64,
                 balance_classes=0, onlyrandBoxes=False, max_object_size=0., n_boxes = 1, use_gt_mask=0, boxrotate=0):
        self.image_path = os.path.join('data','coco','images')
        self.transform = transform
        self.mode = mode
        self.iouThresh = 0.5
        self.dataset = json.load(open(os.path.join('data','pascalVoc','dataset.json'),'r'))
        self.num_data = len(self.dataset['images'])
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.out_img_size = out_img_size
        self.bbox_out_size = bbox_out_size
        #self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.selected_attrs = select_attrs
        self.balance_classes = balance_classes
        self.onlyrandBoxes = onlyrandBoxes
        self.max_object_size = max_object_size

        self.randHFlip = 'Flip' in transform

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        self.num_data = len(self.dataset['images'])

    def preprocess(self):
        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}

        # First remove unwanted splits:
        self.dataset['images'] = [img for img in self.dataset['images'] if img['split'] == self.mode]
        self.valid_ids = [img['filename'].split('.')[0] for img in self.dataset['images']]

        selset = set(self.selected_attrs)
        for i, img in enumerate(self.dataset['images']):
            self.dataset['images'][i]['label'] = np.zeros(max(len(selset),1))
            self.dataset['images'][i]['label'][[self.sattr_to_idx[cls] for cls in img['classes']]] = 1.

        if self.balance_classes:
            self.attToImgId = defaultdict(set)
            for i, img in enumerate(self.dataset['images']):
                if len(img['classes']):
                    for att in img['classes']:
                        self.attToImgId[att].add(i)
                else:
                    self.attToImgId['bg'].add(i)
            self.attToImgId = {k:list(v) for k,v in self.attToImgId.iteritems()}

    def randomBBoxSample(self, index):
        # With 50% chance sample from background or foreground
        # Minimum size
        minLen = 0.3
        maxLen = 0.8
        maxIou = 0.3
        cbboxList = []
        while 1:
            if len(cbboxList) and (random.random()<0.9):
                cbid = random.randrange(len(cbboxList))
                sbox = self.dataset['images'][index]['bboxAnn'][cbid]
                return sbox['bbox'],sbox['box_label'], cbid
            else:
                # sample a random background box
                cbid = None
                tL_x, tL_y = random.uniform(0,1.-minLen-0.01), random.uniform(0,1.-minLen-0.01)
                l_x = random.uniform(minLen, min(1.-tL_x,maxLen))
                l_y = random.uniform(minLen, min(1.-tL_y,maxLen))
                sbox = [tL_x, tL_y, l_x, l_y]
                # Prepare label for this box
                bboxLabel = np.zeros(max(len(self.selected_attrs),1))
                # Test for overlap with foreground objects
                noOverlap = True
                #if len(cbboxList):
                for bb in cbboxList:
                    iou, aInb, bIna = computeIOU(sbox, bb['bbox'])
                    if iou > maxIou or aInb >0.8:
                        noOverlap = False
                    if bIna > 0.8:
                        bboxLabel[self.sattr_to_idx[self.catid2attr[bb['cid']]]] = 1
                if noOverlap:
                    return sbox, bboxLabel, cbid

    def __getitem__(self, index):
        # In this situation ignore index and sample classes uniformly
        if self.balance_classes:
            currCls = random.choice(self.attToImgId.keys())
            index = random.choice(self.attToImgId[currCls])
            cid = [self.sattr_to_idx[currCls]] if currCls != 'bg' else [0]
        else:
            cid = [0]

        image = Image.open(os.path.join(self.image_path,self.dataset['images'][index]['filepath'], self.dataset['images'][index]['filename']))
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')

        bbox, bboxLabel, cbid = self.randomBBoxSample(index)
        label = self.dataset['images'][index]['label']

        # Apply transforms to the image.
        image = self.transform[0](image)
        # Now do the flipping
        if self.randHFlip and random.random()>0.5:
            image = FN.hflip(image)
            bbox[0] = 1.0-(bbox[0]+bbox[2])

        #Convert BBox to actual co-ordinates
        bbox = [int(bc*self.out_img_size) for bc in bbox]
        #print bbox, image.size, cbid
        #assert bbox[3]>0;
        #assert bbox[2]>0;
        #if not ((bbox[0]>=0) and (bbox[0]<128)):
        #    print bbox;
        #    import ipdb;ipdb.set_trace()
        #assert ((bbox[0]>=0) and (bbox[0]<128));
        #assert ((bbox[1]>=0) and (bbox[1]<128))
        #Now obtain the crop
        boxCrop = FN.resized_crop(image, bbox[1], bbox[0], bbox[3],bbox[2], (self.bbox_out_size, self.bbox_out_size))
        # Create Mask
        mask = torch.zeros(1,self.out_img_size,self.out_img_size)
        mask[0,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1.

        return self.transform[-1](image), torch.FloatTensor(label), self.transform[-1](boxCrop), torch.FloatTensor(bboxLabel), mask, torch.IntTensor(bbox), torch.LongTensor(cid)

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        return self.dataset['images'][index]['filename']

    def getcocoid(self, index):
        return self.valid_ids[index]

class MNISTDatasetBBoxSample(Dataset):
    def __init__(self, transform, mode, select_attrs=[], out_img_size=64, bbox_out_size=32, randomrotate=0, scaleRange=[0.1, 0.9], squareAspectRatio=False, use_celeb=False):
        self.image_path = os.path.join('data','mnist')
        self.mode = mode
        self.iouThresh = 0.5
        self.maxDigits= 1
        self.minDigits = 1
        self.use_celeb = use_celeb
        self.scaleRange = scaleRange
        self.squareAspectRatio = squareAspectRatio
        self.nc = 1 if not self.use_celeb else 3
        transList = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(randomrotate,resample=Image.BICUBIC)]#, transforms.ColorJitter(0.5,0.5,0.5,0.3)
        self.digitTransforms = transforms.Compose(transList)
        self.dataset = MNIST(self.image_path,train=True, transform=self.digitTransforms) if not use_celeb else CelebDataset('./data/celebA/images', './data/celebA/list_attr_celeba.txt', self.digitTransforms, mode)
        self.num_data = len(self.dataset)
        self.metadata = {'images':[]}
        self.catid2attr = {}
        self.out_img_size = out_img_size
        self.bbox_out_size = bbox_out_size
        self.selected_attrs = select_attrs

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')

    def preprocess(self):
        for i in xrange(self.num_data):
            n_objects = np.random.randint(self.minDigits, self.maxDigits+1)
            c_digits = 0
            cbboxList = []
            maxIou = 0.1
            c = 0
            while (len(cbboxList) < n_objects) and (c<10):
                c+=1
                tL_x= random.uniform(0,1.-self.scaleRange[0]-0.01)
                tL_y = random.uniform(0,1.-self.scaleRange[0]-0.01)
                l_x = random.uniform(self.scaleRange[0], min(1.-tL_x, self.scaleRange[1]))
                l_y = random.uniform(self.scaleRange[0], min(1.-tL_y, self.scaleRange[1])) if not self.squareAspectRatio else min(1.-tL_y, l_x)
                l_x = l_y if self.squareAspectRatio else l_x
                sbox = [tL_x, tL_y, l_x, l_y]

                noOverlap = True
                for bb in cbboxList:
                    iou, aInb, bIna = computeIOU(sbox, bb)
                    if iou > maxIou or aInb>0.8 or bIna>0.8:
                        noOverlap = False
                        break
                    #if bIna > 0.8:
                    #    bboxLabel[self.sattr_to_idx[self.catid2attr[bb['cid']]]] = 1
                if noOverlap:
                   cbboxList.append(sbox)
            self.metadata['images'].append(cbboxList)


    def __getitem__(self, index):
        # Apply transforms to the image.
        image = torch.FloatTensor(self.nc,self.out_img_size, self.out_img_size).fill_(-1.)
        # Get the individual images.
        randbox = random.randrange(len(self.metadata['images'][index]))
        imglabel = np.zeros(10, dtype=np.int)
        boxlabel = np.zeros(10, dtype=np.int)
        for i,bb in enumerate(self.metadata['images'][index]):
            imid = random.randrange(self.num_data)
            bbox = [int(bc*self.out_img_size) for bc in bb]
            img, label = self.dataset[imid]
            scImg = FN.resize(img,(bbox[3],bbox[2]))
            image[:, bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = FN.normalize(FN.to_tensor(scImg), mean=(0.5,)*self.nc, std=(0.5,)*self.nc)
            #imglabel[label] = 1
            if i == randbox:
                outBox = FN.normalize(FN.to_tensor(FN.resize(scImg, (self.bbox_out_size, self.bbox_out_size))), mean=(0.5,)*self.nc, std=(0.5,)*self.nc)
                mask = torch.zeros(1,self.out_img_size,self.out_img_size)
                mask[0,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1.
                outbbox = bbox
                #boxlabel[label]=1

        #return image[[0,0,0],::], torch.FloatTensor([1]), outBox[[0,0,0],::], torch.FloatTensor([1]), mask, torch.IntTensor(outbbox)
        return image, torch.FloatTensor([1]), outBox, torch.FloatTensor([1]), mask, torch.IntTensor(outbbox)

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        return str(index)


class CocoDataset(Dataset):
    def __init__(self, transform, mode, select_attrs=[], datafile='datasetBoxAnn.json', out_img_size = 128, balance_classes=0):
        self.image_path = os.path.join('data','coco','images')
        self.transform = transform
        self.mode = mode
        self.dataset = json.load(open(os.path.join('data','coco',datafile),'r'))
        self.num_data = len(self.dataset['images'])
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.out_img_size = out_img_size
        self.balance_classes = balance_classes

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        self.num_data = len(self.dataset['images'])

    def preprocess(self):
        for i, attr in enumerate(self.dataset['categories']):
            self.attr2idx[attr['name']] = i
            self.idx2attr[i] = attr['name']
            self.catid2attr[attr['id']] = attr['name']

        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}

        # First remove unwanted splits:
        self.dataset['images'] = [img for img in self.dataset['images'] if img['split'] == self.mode]

        selset = set(self.selected_attrs)
        for i, img in enumerate(self.dataset['images']):
            # Correct BBox for Resize(of smaller edge) and CenterCrop
            fixedbbox = []
            imgSize = self.dataset['images'][i]['imgSize']
            maxSide = np.argmax(imgSize)
            for j in xrange(len(self.dataset['images'][i]['bboxAnn'])):
                cbbox = self.dataset['images'][i]['bboxAnn'][j]
                maxSideLen = int(float(self.out_img_size * imgSize[maxSide]) / (imgSize[1-maxSide]))
                assert(maxSideLen >= self.out_img_size)
                newStartCord = round((maxSideLen - self.out_img_size)/2.)
                boxStart = min( max(cbbox['bbox'][maxSide]*maxSideLen - newStartCord, 0),  self.out_img_size)
                boxEnd =  min(max((cbbox['bbox'][maxSide]+cbbox['bbox'][maxSide+2])*maxSideLen - newStartCord, 0), self.out_img_size)
                length = boxEnd - boxStart
                if length > 5:
                    cbbox['bbox'][maxSide] = float(boxStart)/self.out_img_size
                    cbbox['bbox'][maxSide+2] = float(length)/self.out_img_size
                    if cbbox['bbox'][1-maxSide+2] >= 0.04 and ((length*cbbox['bbox'][1-maxSide+2] * self.out_img_size)> 30.):
                        fixedbbox.append(cbbox)
                        if cbbox['bbox'][0]<0. or cbbox['bbox'][1] < 0. or cbbox['bbox'][0]>1.0 or cbbox['bbox'][1]> 1.0:
                            import ipdb; ipdb.set_trace()
            self.dataset['images'][i]['bboxAnn'] = fixedbbox

            self.dataset['images'][i]['label'] = np.zeros(len(selset))
            self.dataset['images'][i]['label'][[self.sattr_to_idx[self.catid2attr[bb['cid']]] for bb in img['bboxAnn'] if self.catid2attr[bb['cid']] in selset]] = 1.


        # make a list of image id for each class.
        if self.balance_classes:
            self.attToImgId = defaultdict(set)
            for i, img in enumerate(self.dataset['images']):
                classesInImg = [self.catid2attr[bb['cid']] for bb in img['bboxAnn'] if self.catid2attr[bb['cid']] in selset]
                if len(classesInImg):
                    for att in classesInImg:
                        self.attToImgId[att].add(i)
                else:
                    self.attToImgId['bg'].add(i)
            self.attToImgId = {k:list(v) for k,v in self.attToImgId.iteritems()}



    def __getitem__(self, index):
        # In this situation ignore index and sample classes uniformly
        if self.balance_classes:
            currCls = random.choice(self.attToImgId.keys())
            index = random.choice(self.attToImgId[currCls])

        image = Image.open(os.path.join(self.image_path,self.dataset['images'][index]['filepath'], self.dataset['images'][index]['filename']))
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')
        label = self.dataset['images'][index]['label']

        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        return self.dataset['images'][index]['filename']

    def getcocoid(self, index):
        return self.dataset['images'][index]['cocoid']

class CocoMaskDataset(Dataset):
    def __init__(self, transform, mode, select_attrs=[], balance_classes=0, n_masks_perclass=-1):
        self.data_path = os.path.join('data','coco')
        self.transform = transform
        self.mode = mode
        filename = 'instances_train2014.json' if mode=='train' else  'instances_val2014.json'
        self.dataset =  COCOTool(os.path.join(self.data_path, filename))
        self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        valid_ids = []
        for catid in self.dataset.getCatIds(self.selected_attrs):
            valid_ids.extend(self.dataset.getImgIds(catIds=catid))
        self.valid_ids = list(set(valid_ids))
        self.imgId2idx = {imid:i for i,imid in enumerate(self.valid_ids)}
        self.num_data = len(self.valid_ids)
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.nc = 1
        self.balance_classes = balance_classes
        self.n_masks_perclass = n_masks_perclass

        self.preprocess()
        print ('Loaded Mask Data')

    def preprocess(self):
        for atid in self.dataset.cats:
            self.catid2attr[self.dataset.cats[atid]['id']] = self.dataset.cats[atid]['name']

        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}
        self.labels = {}
        self.catsInImg = {}
        self.validAnnotations = {}
        self.imgSizes = {}
        self.validCatIds = self.dataset.getCatIds(self.selected_attrs)

        selset = set(self.selected_attrs)
        for i, imgid in enumerate(self.valid_ids):
            self.labels[i] = np.zeros(len(selset))
            self.labels[i][[self.sattr_to_idx[self.catid2attr[ann['category_id']]] for ann in self.dataset.imgToAnns[imgid] if self.catid2attr[ann['category_id']] in selset]] = 1.
            self.catsInImg[i] = list(set([ann['category_id'] for ann in self.dataset.imgToAnns[imgid] if self.catid2attr[ann['category_id']] in selset]))
            self.imgSizes[i] =  [self.dataset.imgs[imgid]['height'], self.dataset.imgs[imgid]['width']]


        if self.balance_classes:
            self.attToImgId = defaultdict(set)
            for i, imgid in enumerate(self.valid_ids):
                if len(self.catsInImg[i]):
                    for attid in self.catsInImg[i]:
                        self.attToImgId[self.catid2attr[attid]].add(i)
                else:
                    import ipdb; ipdb.set_trace()

            self.attToImgId = {k:list(v) for k,v in self.attToImgId.iteritems()}
            for ann in self.attToImgId:
                    shuffle(self.attToImgId[ann])
            if self.n_masks_perclass >0:
                self.attToImgId = {k:v[:self.n_masks_perclass] for k,v in self.attToImgId.iteritems()}

    def __getitem__(self, index):
        #image = Image.open(os.path.join(self.image_path,self.dataset['images'][index]['filepath'], self.dataset['images'][index]['filename']))
        # In this situation ignore index and sample classes uniformly
        if self.balance_classes:
            currCls = random.choice(self.selected_attrs)
            index = random.choice(self.attToImgId[currCls])

        maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
        label = np.zeros(len(self.selected_attrs))
        if len(self.catsInImg[index]):
            # Randomly sample an annotation
            currObjId = random.choice(self.catsInImg[index])
            for ann in self.dataset.loadAnns(self.dataset.getAnnIds(self.valid_ids[index], currObjId)):
                cm = self.dataset.annToMask(ann)
                maskTotal[:cm.shape[0],:cm.shape[1]] += cm
            label[self.sattr_to_idx[self.catid2attr[currObjId]]] = 1.

        mask = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]

        return mask, torch.FloatTensor(label)

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        return self.dataset['images'][index]['filename']

    def getbyIdAndclass(self, imgid, cls, hflip=0):
        if (imgid not in self.imgId2idx):
            maskTotal = np.zeros((128,128))
        else:
            index= self.imgId2idx[imgid]
            catId = self.dataset.getCatIds(cls)
            maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
            if len(self.catsInImg[index]) and (catId[0] in self.catsInImg[index]):
                # Randomly sample an annotation
                for ann in self.dataset.loadAnns(self.dataset.getAnnIds(self.valid_ids[index], catId)):
                    cm = self.dataset.annToMask(ann)
                    maskTotal[:cm.shape[0],:cm.shape[1]] += cm
            if hflip:
                maskTotal = maskTotal[:,::-1]

        mask = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]

        return mask

    def getbyClass(self, cls):
        allMasks = []
        for c in cls:
            curr_obj = self.selected_attrs[c]
            catId = self.dataset.getCatIds(curr_obj)
            index = random.choice(self.attToImgId[curr_obj])
            maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
            if len(self.catsInImg[index]):
                # Randomly sample an annotation
                for ann in self.dataset.loadAnns(self.dataset.getAnnIds(self.valid_ids[index], catId)):
                    cm = self.dataset.annToMask(ann)
                    maskTotal[:cm.shape[0],:cm.shape[1]] += cm
            maskTotal = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]
            allMasks.append(maskTotal[None,::])

        return torch.cat(allMasks,dim=0)

    def getbyIdAndclassBatch(self, imgid, cls, hFlips = None):
        allMasks = []
        for i,c in enumerate(cls):
            curr_obj = self.selected_attrs[c]
            catId = self.dataset.getCatIds(curr_obj)
            if (imgid[i] not in self.imgId2idx):
                maskTotal = np.zeros((128,128))
            else:
                index = self.imgId2idx[imgid[i]]
                maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
                if len(self.catsInImg[index]) and (catId[0] in self.catsInImg[index]):
                    # Randomly sample an annotation
                    for ann in self.dataset.loadAnns(self.dataset.getAnnIds(imgid[i], catId)):
                        cm = self.dataset.annToMask(ann)
                        maskTotal[:cm.shape[0],:cm.shape[1]] += cm
                if (hFlips is not None) and hFlips[i] == 1:
                    maskTotal = maskTotal[:,::-1]
            maskTotal = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]
            allMasks.append(maskTotal[None,::])

        return torch.cat(allMasks,dim=0)

class SDI_MaskDataset(Dataset):
    def __init__(self, transform, mode, select_attrs=[], balance_classes=0, n_masks_perclass=-1):
        self.data_path = os.path.join('data','coco')
        self.transform = transform
        self.mode = mode
        filename = 'instances_train2014.json' if mode=='train' else  'instances_val2014.json'
        self.dataset =  COCOTool(os.path.join(self.data_path, filename))
        self.SDI_filelist = open(os.path.join(self.data_path,'SDI_img_list_val.txt'),'r').read().splitlines()
        self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.valid_ids = []
        self.maskToObject = {1:'airplane', 2:'bicycle', 3:'bird', 4:'boat', 5:'bottle', 6:'bus', 7:'car' , 8:'cat', 9:'chair', 10:'cow',
                             11:'dining table', 12:'dog', 13:'horse', 14:'motorcycle', 15:'person', 16:'potted plant', 17:'sheep', 18:'couch',
                             19:'train', 20:'tv'}
        self.objectToMask = {self.maskToObject[v]:v for v in self.maskToObject}
        self.seg_directory = os.path.join(self.data_path, 'SDI_segmentation')

        #remove irrelavant results
        self.imgId2idx = {}
        for i, fname in enumerate(self.SDI_filelist):
            cocoid = int(fname.split('.')[0].split('_')[-1])
            self.valid_ids.append(cocoid)
            self.imgId2idx[cocoid] = i
        self.num_data = len(self.valid_ids)
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.nc = 1
        self.balance_classes = balance_classes
        self.n_masks_perclass = n_masks_perclass

        self.preprocess()
        print ('Loaded Mask Data')

    def preprocess(self):
        for atid in self.dataset.cats:
            self.catid2attr[self.dataset.cats[atid]['id']] = self.dataset.cats[atid]['name']

        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}
        self.labels = {}
        self.catsInImg = {}
        self.validAnnotations = {}
        self.imgSizes = {}
        self.validCatIds = self.dataset.getCatIds(self.selected_attrs)

        #self.imgToAnns = defaultdict(list)
        #self.imgToCatToAnns = defaultdict(dict)
        #for i,ann in enumerate(self.mRCNN_results):
        #    self.imgToAnns[ann['image_id']].append(i)
        #    if self.catid2attr[ann['category_id']] not in self.imgToCatToAnns[ann['image_id']]:
        #        self.imgToCatToAnns[ann['image_id']][self.catid2attr[ann['category_id']]] = []
        #    self.imgToCatToAnns[ann['image_id']][self.catid2attr[ann['category_id']]].append(i)

        selset = set(self.selected_attrs)
        for i, imgid in enumerate(self.valid_ids):
            #self.labels[i] = np.zeros(len(selset))
            #self.labels[i][[self.sattr_to_idx[self.catid2attr[ann['category_id']]] for ann in self.dataset.imgToAnns[imgid] if self.catid2attr[ann['category_id']] in selset]] = 1.
            #self.catsInImg[i] = list(set([ann['category_id'] for ann in self.dataset.imgToAnns[imgid] if self.catid2attr[ann['category_id']] in selset]))
            self.imgSizes[i] =  [self.dataset.imgs[imgid]['width'],self.dataset.imgs[imgid]['height']]

        #if self.balance_classes:
        #    self.attToImgId = defaultdict(set)
        #    for i, imgid in enumerate(self.valid_ids):
        #        if len(self.catsInImg[i]):
        #            for attid in self.catsInImg[i]:
        #                self.attToImgId[self.catid2attr[attid]].add(i)
        #        else:
        #            import ipdb; ipdb.set_trace()

        #    self.attToImgId = {k:list(v) for k,v in self.attToImgId.iteritems()}
        #    for ann in self.attToImgId:
        #            shuffle(self.attToImgId[ann])
        #    if self.n_masks_perclass >0:
        #        self.attToImgId = {k:v[:self.n_masks_perclass] for k,v in self.attToImgId.iteritems()}

    def __getitem__(self, index):
        #image = Image.open(os.path.join(self.image_path,self.dataset['images'][index]['filepath'], self.dataset['images'][index]['filename']))
        # In this situation ignore index and sample classes uniformly
        assert 0, 'This is not implemented'
        if self.balance_classes:
            currCls = random.choice(self.selected_attrs)
            index = random.choice(self.attToImgId[currCls])

        maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
        label = np.zeros(len(self.selected_attrs))
        if len(self.catsInImg[index]):
            # Randomly sample an annotation
            currObjId = random.choice(self.catsInImg[index])
            for ann in self.dataset.loadAnns(self.dataset.getAnnIds(self.valid_ids[index], currObjId)):
                cm = self.dataset.annToMask(ann)
                maskTotal[:cm.shape[0],:cm.shape[1]] += cm
            label[self.sattr_to_idx[self.catid2attr[currObjId]]] = 1.

        mask = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]

        return mask, torch.FloatTensor(label)

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        return self.dataset['images'][index]['filename']

    def getbyIdAndclass(self, imgid, cls, hflip=0):
        if (imgid not in self.imgId2idx) or (cls == 'bg'):
            maskTotal = np.zeros((128,128))
        else:
            index= self.imgId2idx[imgid]
            catId = self.dataset.getCatIds(cls)
            segImg = Image.open(os.path.join(self.seg_directory, self.SDI_filelist[index]))
            oImg_sz = self.imgSizes[index]
            maxSide = np.argmax(oImg_sz)
            assert(segImg.size[0] == segImg.size[1])
            crop_sizes = [0.,0.]
            crop_sizes[maxSide] = segImg.size[0]
            crop_sizes[1-maxSide] = int(oImg_sz[1-maxSide] * (float(segImg.size[0])/float(oImg_sz[maxSide])))
            segImg = segImg.crop([0,0,crop_sizes[0],crop_sizes[1]])
            maskTotal = (np.array(segImg) == self.objectToMask[cls]).astype(np.float)
            #if len(self.catsInImg[index]) and (catId[0] in self.catsInImg[index]) and (cls in self.imgToCatToAnns[imgid]):
            #    for annIndex in self.imgToCatToAnns[imgid][cls]:
            #        ann = self.mRCNN_results[annIndex]
            #        cm = self.dataset.annToMask(ann)
            #        maskTotal[:cm.shape[0],:cm.shape[1]] += cm
            if hflip:
                maskTotal = maskTotal[:,::-1]

        mask = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]

        return mask

    def getbyClass(self, cls):
        assert 0, 'This is not implemented'
        #allMasks = []
        #for c in cls:
        #    curr_obj = self.selected_attrs[c]
        #    catId = self.dataset.getCatIds(curr_obj)
        #    index = random.choice(self.attToImgId[curr_obj])
        #    maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
        #    if len(self.catsInImg[index]):
        #        # Randomly sample an annotation
        #        for ann in self.dataset.loadAnns(self.dataset.getAnnIds(self.valid_ids[index], catId)):
        #            cm = self.dataset.annToMask(ann)
        #            maskTotal[:cm.shape[0],:cm.shape[1]] += cm
        #    maskTotal = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]
        #    allMasks.append(maskTotal[None,::])

        #return torch.cat(allMasks,dim=0)

    def getbyIdAndclassBatch(self, imgid, cls):
        assert 0, 'This is not implemented'
        #allMasks = []
        #for i,c in enumerate(cls):
        #    curr_obj = self.selected_attrs[c]
        #    catId = self.dataset.getCatIds(curr_obj)
        #    index = self.imgId2idx[imgid[i]]
        #    maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
        #    if len(self.catsInImg[index]) and (catId in self.catsInImg[index]):
        #        # Randomly sample an annotation
        #        for ann in self.dataset.loadAnns(self.dataset.getAnnIds(imgid[i], catId)):
        #            cm = self.dataset.annToMask(ann)
        #            maskTotal[:cm.shape[0],:cm.shape[1]] += cm
        #    maskTotal = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]
        #    allMasks.append(maskTotal[None,::])

        #return torch.cat(allMasks,dim=0)


class MRCNN_MaskDataset(Dataset):
    def __init__(self, transform, mode, select_attrs=[], balance_classes=0, n_masks_perclass=-1):
        self.data_path = os.path.join('data','coco')
        self.transform = transform
        self.mode = mode
        filename = 'instances_train2014.json' if mode=='train' else  'instances_val2014.json'
        self.dataset =  COCOTool(os.path.join(self.data_path, filename))
        self.mRCNN_results = json.load(open(os.path.join(self.data_path,'maskRCNN_masks_X-101-64x4d-FPN_mAp37p5_coco_2014_minival_results.json'),'r'))
        self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        valid_ids = []
        val_cat_ids = set(self.dataset.getCatIds(self.selected_attrs))
        #remove irrelavant results
        self.mRCNN_results = [ann for ann in self.mRCNN_results if ann['category_id'] in val_cat_ids]
        for ann in self.mRCNN_results:
            valid_ids.append(ann['image_id'])
        self.valid_ids = list(set(valid_ids))
        self.imgId2idx = {imid:i for i,imid in enumerate(self.valid_ids)}
        self.num_data = len(self.valid_ids)
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.nc = 1
        self.balance_classes = balance_classes
        self.n_masks_perclass = n_masks_perclass

        self.preprocess()
        print ('Loaded Mask Data')

    def preprocess(self):
        for atid in self.dataset.cats:
            self.catid2attr[self.dataset.cats[atid]['id']] = self.dataset.cats[atid]['name']

        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}
        self.labels = {}
        self.catsInImg = {}
        self.validAnnotations = {}
        self.imgSizes = {}
        self.validCatIds = self.dataset.getCatIds(self.selected_attrs)

        self.imgToAnns = defaultdict(list)
        self.imgToCatToAnns = defaultdict(dict)
        for i,ann in enumerate(self.mRCNN_results):
            self.imgToAnns[ann['image_id']].append(i)
            if self.catid2attr[ann['category_id']] not in self.imgToCatToAnns[ann['image_id']]:
                self.imgToCatToAnns[ann['image_id']][self.catid2attr[ann['category_id']]] = []
            self.imgToCatToAnns[ann['image_id']][self.catid2attr[ann['category_id']]].append(i)

        selset = set(self.selected_attrs)
        for i, imgid in enumerate(self.valid_ids):
            self.labels[i] = np.zeros(len(selset))
            self.labels[i][[self.sattr_to_idx[self.catid2attr[ann['category_id']]] for ann in self.dataset.imgToAnns[imgid] if self.catid2attr[ann['category_id']] in selset]] = 1.
            self.catsInImg[i] = list(set([ann['category_id'] for ann in self.dataset.imgToAnns[imgid] if self.catid2attr[ann['category_id']] in selset]))
            self.imgSizes[i] =  [self.dataset.imgs[imgid]['height'], self.dataset.imgs[imgid]['width']]


        if self.balance_classes:
            self.attToImgId = defaultdict(set)
            for i, imgid in enumerate(self.valid_ids):
                if len(self.catsInImg[i]):
                    for attid in self.catsInImg[i]:
                        self.attToImgId[self.catid2attr[attid]].add(i)
                else:
                    import ipdb; ipdb.set_trace()

            self.attToImgId = {k:list(v) for k,v in self.attToImgId.iteritems()}
            for ann in self.attToImgId:
                    shuffle(self.attToImgId[ann])
            if self.n_masks_perclass >0:
                self.attToImgId = {k:v[:self.n_masks_perclass] for k,v in self.attToImgId.iteritems()}

    def __getitem__(self, index):
        #image = Image.open(os.path.join(self.image_path,self.dataset['images'][index]['filepath'], self.dataset['images'][index]['filename']))
        # In this situation ignore index and sample classes uniformly
        if self.balance_classes:
            currCls = random.choice(self.selected_attrs)
            index = random.choice(self.attToImgId[currCls])

        maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
        label = np.zeros(len(self.selected_attrs))
        if len(self.catsInImg[index]):
            # Randomly sample an annotation
            currObjId = random.choice(self.catsInImg[index])
            for ann in self.dataset.loadAnns(self.dataset.getAnnIds(self.valid_ids[index], currObjId)):
                cm = self.dataset.annToMask(ann)
                maskTotal[:cm.shape[0],:cm.shape[1]] += cm
            label[self.sattr_to_idx[self.catid2attr[currObjId]]] = 1.

        mask = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]

        return mask, torch.FloatTensor(label)

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        return self.dataset['images'][index]['filename']

    def getbyIdAndclass(self, imgid, cls, hflip=0):
        if (imgid not in self.imgId2idx) or (cls == 'bg'):
            maskTotal = np.zeros((128,128))
        else:
            index= self.imgId2idx[imgid]
            catId = self.dataset.getCatIds(cls)
            maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
            if len(self.catsInImg[index]) and (catId[0] in self.catsInImg[index]) and (cls in self.imgToCatToAnns[imgid]):
                # Randomly sample an annotation
                for annIndex in self.imgToCatToAnns[imgid][cls]:
                    ann = self.mRCNN_results[annIndex]
                    cm = self.dataset.annToMask(ann)
                    maskTotal[:cm.shape[0],:cm.shape[1]] += cm
            if hflip:
                maskTotal = maskTotal[:,::-1]

        mask = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]

        return mask

    def getbyClass(self, cls):
        allMasks = []
        for c in cls:
            curr_obj = self.selected_attrs[c]
            catId = self.dataset.getCatIds(curr_obj)
            index = random.choice(self.attToImgId[curr_obj])
            maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
            if len(self.catsInImg[index]):
                # Randomly sample an annotation
                for ann in self.dataset.loadAnns(self.dataset.getAnnIds(self.valid_ids[index], catId)):
                    cm = self.dataset.annToMask(ann)
                    maskTotal[:cm.shape[0],:cm.shape[1]] += cm
            maskTotal = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]
            allMasks.append(maskTotal[None,::])

        return torch.cat(allMasks,dim=0)

    def getbyIdAndclassBatch(self, imgid, cls):
        allMasks = []
        for i,c in enumerate(cls):
            curr_obj = self.selected_attrs[c]
            catId = self.dataset.getCatIds(curr_obj)
            index = self.imgId2idx[imgid[i]]
            maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
            if len(self.catsInImg[index]) and (catId in self.catsInImg[index]):
                # Randomly sample an annotation
                for ann in self.dataset.loadAnns(self.dataset.getAnnIds(imgid[i], catId)):
                    cm = self.dataset.annToMask(ann)
                    maskTotal[:cm.shape[0],:cm.shape[1]] += cm
            maskTotal = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]
            allMasks.append(maskTotal[None,::])

        return torch.cat(allMasks,dim=0)


class PascalMaskDataset(Dataset):
    def __init__(self, transform, mode, select_attrs=[], balance_classes=0, n_masks_perclass=-1):
        self.data_path = os.path.join('data','pascalVoc')
        self.transform = transform
        self.mode = mode
        self.dataset = np.load(os.path.join(self.data_path, 'maskdata.npy')).item()
        self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.valid_ids = self.dataset.keys()
        self.num_data = len(self.valid_ids)
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.nc = 1
        self.balance_classes = 0 #balance_classes
        self.n_masks_perclass = n_masks_perclass

        self.preprocess()
        print ('Loaded Mask Data')

    def preprocess(self):
        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}
        self.labels = {}
        self.catsInImg = {}
        self.validAnnotations = {}
        self.imgSizes = {}
        self.maskToObject = {1:'airplane', 2:'bicycle', 3:'bird', 4:'boat', 5:'bottle', 6:'bus', 7:'car' , 8:'cat', 9:'chair', 10:'cow',
                             11:'dining table', 12:'dog', 13:'horse', 14:'motorcycle', 15:'person', 16:'potted plant', 17:'sheep', 18:'couch',
                             19:'train', 20:'tv'}

        self.objectToMask = {self.maskToObject[v]:v for v in self.maskToObject}

        selset = set(self.selected_attrs)
        for i, imgid in enumerate(self.valid_ids):
            self.labels[i] = np.zeros(len(selset))
            self.labels[i][[self.sattr_to_idx[ann] for ann in self.dataset[imgid]['label'] if ann in selset]] = 1.
            self.catsInImg[i] = list(set([ann for ann in self.dataset[imgid]['label'] if ann in selset]))

        self.attToImgId = defaultdict(set)
        for i, imgid in enumerate(self.valid_ids):
            if len(self.catsInImg[i]):
                for att in self.catsInImg[i]:
                    self.attToImgId[att].add(i)
            #else:
            #    import ipdb; ipdb.set_trace()
        self.attToImgId = {k:list(v) for k,v in self.attToImgId.iteritems()}
        for ann in self.attToImgId:
                shuffle(self.attToImgId[ann])
        if self.n_masks_perclass >0:
            self.attToImgId = {k:v[:self.n_masks_perclass] for k,v in self.attToImgId.iteritems()}


    def __getitem__(self, index):
        #image = Image.open(os.path.join(self.image_path,self.dataset['images'][index]['filepath'], self.dataset['images'][index]['filename']))
        # In this situation ignore index and sample classes uniformly
        if self.balance_classes:
            currCls = random.choice(self.selected_attrs)
            index = random.choice(self.attToImgId[currCls])

        label = np.zeros(len(self.selected_attrs))
        if len(self.catsInImg[index]):
            # Randomly sample an annotation
            currObj = random.choice(self.catsInImg[index])
            maskTotal = (self.dataset[self.valid_ids[index]]['mask'] == self.objectToMask[currObj]).astype(np.float)
            label[self.sattr_to_idx[currObj]] = 1.
        else:
            print 'No obj in %s'%self.valid_ids[index]
            maskTotal = np.zeros((128, 128))

        mask = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]

        return mask, torch.FloatTensor(label)

    def getbyClass(self, cls):
        allMasks = []
        for c in cls:
            curr_obj = self.selected_attrs[c]
            index = random.choice(self.attToImgId[curr_obj])
            maskTotal = (self.dataset[self.valid_ids[index]]['mask'] == self.objectToMask[curr_obj]).astype(np.float)
            maskTotal = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]
            allMasks.append(maskTotal[None,::])

        return torch.cat(allMasks,dim=0)

    def getbyIdAndclass(self, imgid, cls):
        if imgid not in self.valid_ids:
            print 'Specified coco id not found'
            return
        index= self.valid_ids.index(imgid)
        maskTotal = (self.dataset[self.valid_ids[index]]['mask'] == self.objectToMask[cls]).astype(np.float)
        maskTotal = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]
        return maskTotal

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        return self.valid_ids[index]


class CelebDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform, mode, select_attrs=[]):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.lines = open(metadata_path, 'r').readlines()
        self.num_data = int(self.lines[0])
        self.attr2idx = {}
        self.idx2attr = {}
        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'] if select_attrs== [] else select_attrs

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

    def preprocess(self):
        attrs = self.lines[1].split()
        for i, attr in enumerate(attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr

        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []

        lines = self.lines[2:]
        random.shuffle(lines)   # random shuffling
        for i, line in enumerate(lines):

            splits = line.split()
            filename = splits[0]
            values = splits[1:]

            label = []
            for idx, value in enumerate(values):
                attr = self.idx2attr[idx]

                if attr in self.selected_attrs:
                    if value == '1':
                        label.append(1)
                    else:
                        label.append(0)

            if (i+1) < 2000:
                self.test_filenames.append(filename)
                self.test_labels.append(label)
            else:
                self.train_filenames.append(filename)
                self.train_labels.append(label)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(os.path.join(self.image_path, self.train_filenames[index]))
            label = self.train_labels[index]
        elif self.mode in ['test']:
            image = Image.open(os.path.join(self.image_path, self.test_filenames[index]))
            label = self.test_labels[index]

        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_data

    def getfilename(self, index):
        if self.mode == 'train':
            return self.train_filenames[index]
        else:
            return self.test_filenames[index]


def get_loader(image_path, metadata_path, crop_size, image_size, batch_size, dataset='CelebA', mode='train',
               select_attrs=[], datafile='datasetBoxAnn.json', bboxLoader=False, bbox_size = 64,
               randomrotate=0, randomscale=(0.5, 0.5), loadMasks=False, balance_classes=0, onlyrandBoxes=False,
               max_object_size=0., n_masks=-1, imagenet_norm=False, use_gt_mask = False, n_boxes = 1, square_resize = 0,
               filter_by_mincooccur = -1., only_indiv_occur = 0, augmenter_mode = 0):
    """Build and return data loader."""

    transList = [transforms.Resize(image_size if not square_resize else [image_size, image_size]), transforms.CenterCrop(image_size)] if not loadMasks else [transforms.Resize(image_size if not square_resize else [image_size, image_size], interpolation=Image.NEAREST), transforms.RandomCrop(image_size)]
    if mode == 'train':
        transList.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transList.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if imagenet_norm:
        transList[-1] = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    if loadMasks:
        transform = transforms.Compose(transList[:-2])
    elif bboxLoader:
        # Split the transforms into 3 parts.
        # First is applied on the entire image before cropping
        # Second part consists of random augments which needs special handling
        # second is applied to convert image to tensor applied sperately to image and crop
        transform = [transforms.Compose(transList[:2]), 'Flip' if mode=='train' else None, transforms.Compose(transList[-2:])]
    else:
        transform = transforms.Compose(transList)

    if loadMasks:
        if dataset == 'coco':
            dataset = CocoMaskDataset(transform, mode, select_attrs=select_attrs, balance_classes=balance_classes,
                    n_masks_perclass=n_masks)
        elif dataset == 'mrcnn':
            dataset = MRCNN_MaskDataset(transform, mode, select_attrs=select_attrs, balance_classes=balance_classes,
                    n_masks_perclass=n_masks)
        elif dataset == 'sdi':
            dataset = SDI_MaskDataset(transform, mode, select_attrs=select_attrs, balance_classes=balance_classes,
                    n_masks_perclass=n_masks)
        elif dataset == 'pascal':
            dataset = PascalMaskDataset(transform, mode, select_attrs=select_attrs, balance_classes=balance_classes,
                    n_masks_perclass=n_masks)
    else:
        if dataset == 'CelebA':
            dataset = CelebDataset(image_path, metadata_path, transform, mode, select_attrs=select_attrs)
        elif dataset == 'RaFD':
            dataset = ImageFolder(image_path, transform)
        elif dataset == 'coco':
            if bboxLoader:
                dataset = CocoDatasetBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                        balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                        use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes, square_resize = square_resize,
                        filter_by_mincooccur = filter_by_mincooccur, only_indiv_occur = only_indiv_occur, augmenter_mode = augmenter_mode)
            else:
                dataset = CocoDataset(transform, mode, select_attrs=select_attrs, datafile=datafile,
                        out_img_size=image_size, balance_classes=balance_classes)
        elif dataset == 'places2':
            dataset = Places2DatasetBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                    balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                    use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'ade20k':
            dataset = ADE20k(transform, mode, select_attrs, image_size, bbox_size, max_object_size=max_object_size,
                    use_gt_mask = use_gt_mask, boxrotate= randomrotate, n_boxes = n_boxes, square_resize = square_resize)
        elif dataset == 'flickrlogo':
            dataset = FlickrLogoBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                      balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                      use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'belgalogo':
            dataset = BelgaLogoBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                      balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                      use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'outofcontext':
            dataset = OutofContextBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                      balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                      use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'unrel':
            dataset = UnrelBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                      balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                      use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'pascal':
            if bboxLoader:
                dataset = PascalDatasetBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                        balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                        use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'mnist':
            dataset = MNISTDatasetBBoxSample(transform, mode, select_attrs, image_size, bbox_size,
                    randomrotate=randomrotate, scaleRange=randomscale)
        elif dataset == 'celebbox':
            dataset = MNISTDatasetBBoxSample(transform, mode, select_attrs, image_size, bbox_size,
                    randomrotate=randomrotate, scaleRange=randomscale, squareAspectRatio=True, use_celeb=True)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16 if not loadMasks else 2 if image_size==32 else 6, pin_memory=True)
    return data_loader

def get_dataset(image_path, metadata_path, crop_size, image_size, dataset='CelebA', split='train', select_attrs=[],
                datafile='datasetBoxAnn.json', bboxLoader=False, bbox_size = 64, randomrotate=0,
                randomscale=(0.5, 0.5), loadMasks=False, balance_classes=0, onlyrandBoxes=False, max_object_size=0.,
                n_masks=-1, imagenet_norm=False, use_gt_mask = False, mode='test', n_boxes = 1, square_resize = 0,
                filter_by_mincooccur = -1., only_indiv_occur = 0, augmenter_mode = 0):
    """Build and return data loader."""

    transList = [transforms.Resize(image_size if not square_resize else [image_size, image_size]), transforms.CenterCrop(image_size)] if not loadMasks else [transforms.Resize(image_size if not square_resize else [image_size, image_size], interpolation=Image.NEAREST), transforms.RandomCrop(image_size)]
    if mode == 'train':
        transList.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        if loadMasks:
            transList[-1] = transforms.CenterCrop(image_size)
        transList.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if imagenet_norm:
        transList[-1] = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

    if loadMasks:
        transform = transforms.Compose(transList[:-2])
    elif bboxLoader:
        # Split the transforms into 3 parts.
        # First is applied on the entire image before cropping
        # Second part consists of random augments which needs special handling
        # second is applied to convert image to tensor applied sperately to image and crop
        transform = [transforms.Compose(transList[:2]), 'Flip' if mode=='train' else None, transforms.Compose(transList[-2:])]
    else:
        transform = transforms.Compose(transList)

    if loadMasks:
        if dataset == 'coco':
            dataset = CocoMaskDataset(transform, split, select_attrs=select_attrs, balance_classes=balance_classes, n_masks_perclass=n_masks)
        elif dataset == 'mrcnn':
            dataset = MRCNN_MaskDataset(transform, split, select_attrs=select_attrs, balance_classes=balance_classes,
                    n_masks_perclass=n_masks)
        elif dataset == 'sdi':
            dataset = SDI_MaskDataset(transform, mode, select_attrs=select_attrs, balance_classes=balance_classes,
                    n_masks_perclass=n_masks)
        elif dataset == 'pascal':
            dataset = PascalMaskDataset(transform, split, select_attrs=select_attrs, balance_classes=balance_classes, n_masks_perclass=n_masks)
    else:
        if dataset == 'CelebA':
            dataset = CelebDataset(image_path, metadata_path, transform, split)
        elif dataset == 'RaFD':
            dataset = ImageFolder(image_path, transform)
        elif dataset == 'coco':
            if bboxLoader:
                dataset = CocoDatasetBBoxSample(transform, split, select_attrs, datafile, image_size, bbox_size,
                                                balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes,
                                                max_object_size=max_object_size, use_gt_mask = use_gt_mask,
                                                boxrotate= randomrotate, n_boxes = n_boxes, square_resize = square_resize,
                                                filter_by_mincooccur = filter_by_mincooccur, only_indiv_occur = only_indiv_occur,
                                                augmenter_mode = augmenter_mode)
            else:
                dataset = CocoDataset(transform, split, select_attrs=select_attrs, datafile=datafile,
                                      out_img_size=image_size, balance_classes=balance_classes)
        elif dataset == 'places2':
            dataset = Places2DatasetBBoxSample(transform, split, select_attrs, datafile, image_size, bbox_size,
                    balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                    use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'ade20k':
            dataset = ADE20k(transform, split, select_attrs, image_size, bbox_size, max_object_size=max_object_size,
                    use_gt_mask = use_gt_mask, boxrotate= randomrotate, n_boxes = n_boxes, square_resize = square_resize)
        elif dataset == 'flickrlogo':
                dataset = FlickrLogoBBoxSample(transform, split, select_attrs, datafile, image_size, bbox_size,
                        balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                        use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'outofcontext':
            dataset = OutofContextBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                      balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                      use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'unrel':
            dataset = UnrelBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                      balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                      use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'belgalogo':
                dataset = BelgaLogoBBoxSample(transform, split, select_attrs, datafile, image_size, bbox_size,
                        balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                        use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'pascal':
            if bboxLoader:
                dataset = PascalDatasetBBoxSample(transform, split, select_attrs, datafile, image_size, bbox_size,
                                                  balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes,
                                                  max_object_size=max_object_size, use_gt_mask = use_gt_mask,
                                                  boxrotate= randomrotate, n_boxes = n_boxes)
        elif dataset == 'mnist':
            dataset = MNISTDatasetBBoxSample(transform, split, select_attrs, image_size, bbox_size, randomrotate=randomrotate, scaleRange=randomscale)
        elif dataset == 'celebbox':
            dataset = MNISTDatasetBBoxSample(transform, split, select_attrs, image_size, bbox_size, randomrotate=randomrotate, scaleRange=randomscale, squareAspectRatio=True, use_celeb=True)

    return dataset
