### Analyze object sizes ###
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from collections import Counter

data = json.load(open('/BS/rshetty-wrk/archive00/data/cocoDataStuff/datasetBoxAnn.json','r'))
catid2attr = {}
select_attr_list = set(['person', 'book', 'car', 'bird', 'chair'])

for i, attr in enumerate(data['categories']):
    catid2attr[attr['id']] = attr['name']

objectSizes = defaultdict(list)
for img in data['images']:
    for bb in img['bboxAnn']:
        if catid2attr[bb['cid']] in select_attr_list:
            objectSizes[catid2attr[bb['cid']]].append(bb['bbox'][2]*bb['bbox'][3])

colors= ['r','g','b','k','c']
for i,k in enumerate(objectSizes):
    cnt, cent = np.histogram(objectSizes[k],bins=bins/100.)
    plt.loglog((cent[:-1]+cent[1:])/2.*100., cnt, colors[i])
plt.xlabel('Percentage area of the image occupied by the object')
plt.ylabel('Number of Instances of the object')
plt.legend(objectSizes.keys())
plt.show()


sizeOfLargestObj = []
for img in data['images']:
    if len(img['bboxAnn']):
        sizeOfLargestObj.append(max([bb['bbox'][2]*bb['bbox'][3] for bb in img['bboxAnn']]))





#######################################################################
ipdb> plt.imshow(((fak_x.numpy()[0,[0,1,2],:,:].transpose(1,2,0)+1.0)*255./2.0).astype(np.uint8)); plt.show()
*** NameError: name 'fak_x' is not defined
ipdb> plt.imshow(((fake_x.numpy()[0,[0,1,2],:,:].transpose(1,2,0)+1.0)*255./2.0).astype(np.uint8)); plt.show()
*** AttributeError: 'Variable' object has no attribute 'numpy'
ipdb> plt.imshow(((fake_x.data.numpy()[0,[0,1,2],:,:].transpose(1,2,0)+1.0)*255./2.0).astype(np.uint8)); plt.show()
*** RuntimeError: can't convert CUDA tensor to numpy (it doesn't support GPU arrays). Use .cpu() to move the tensor to host memory first.
ipdb> plt.imshow(((fake_x.data.cpu().numpy()[0,[0,1,2],:,:].transpose(1,2,0)+1.0)*255./2.0).astype(np.uint8)); plt.show()
<matplotlib.image.AxesImage object at 0x7fb660035a50>
ipdb> plt.imshow(((fake_x.data.cpu().numpy()[1,[0,1,2],:,:].transpose(1,2,0)+1.0)*255./2.0).astype(np.uint8)); plt.show()
<matplotlib.image.AxesImage object at 0x7fb649c9e8d0>
ipdb> plt.imshow(((mask.data.cpu().numpy()[1,[0,1,2],:,:].transpose(1,2,0))*255.).astype(np.uint8)); plt.show()
<matplotlib.image.AxesImage object at 0x7fb649b966d0>
ipdb> plt.imshow(((1-mask.data.cpu().numpy()[1,[0,1,2],:,:].transpose(1,2,0))*255.).astype(np.uint8)); plt.show()
<matplotlib.image.AxesImage object at 0x7fb649ae8ad0>
ipdb> plt.imshow(((diffimg.data.cpu().numpy()[1,[0,1,2],:,:].transpose(1,2,0)+1.0)*255./2.0).astype(np.uint8)); plt.show()
<matplotlib.image.AxesImage object at 0x7fb649145bd0>
ipdb> plt.imshow(((fake_x.data.cpu().numpy()[2,[0,1,2],:,:].transpose(1,2,0)+1.0)*255./2.0).astype(np.uint8)); plt.show()
<matplotlib.image.AxesImage object at 0x7fb64908acd0>
ipdb> plt.imshow(((diffimg.data.cpu().numpy()[2,[0,1,2],:,:].transpose(1,2,0)+1.0)*255./2.0).astype(np.uint8)); plt.show()
<matplotlib.image.AxesImage object at 0x7fb648fcccd0>
ipdb> plt.imshow(((1-mask.data.cpu().numpy()[2,[0,1,2],:,:].transpose(1,2,0))*255.).astype(np.uint8)); plt.show()
<matplotlib.image.AxesImage object at 0x7fb648f1e410>




#
from utils.data_loader_stargan import get_dataset
import matplotlib.pyplot as plt
import numpy as np

dataset = get_dataset(config.celebA_image_path, config.metadata_path, config.celebA_crop_size, config.image_size, config.dataset, config.mode, select_attrs=config.selected_attrs, datafile=config.datafile,bboxLoader=config.train_boxreconst)
img, imgLab, boxImg, boxLab, mask = dataset[0]

plt.figure();plt.imshow(((img.numpy()[[0,1,2],:,:].transpose(1,2,0)+1.0)*255./2.0).astype(np.uint8)); plt.figure();plt.imshow(((boxImg.numpy()[[0,1,2],:,:].transpose(1,2,0)+1.0)*255./2.0).astype(np.uint8));plt.figure(); plt.imshow(((mask.numpy()[[0,0,0],:,:].transpose(1,2,0)+1.0)*255./2.0).astype(np.uint8));plt.figure();plt.imshow((((img*mask).numpy()[[0,1,2],:,:].transpose(1,2,0)+1.0)*255./2.0).astype(np.uint8));plt.show()




###----------------------------------------------------------------------------------------------------------------

import numpy as np
from collections import defaultdict
from tqdm import tqdm


pwd
imgList = open('trainval.txt').read().splitlines()
imgListVal = open('../Segmentation/val.txt').read().splitlines()
trainList = set(imgList) - set(imgListVal)
allImgs = {}
classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train', 'bottle', 'couch', 'dining table', 'potted plant', 'chair', 'tv'
]
clsToPascalCls = {c:c for c in classes}
clsToPascalCls['dining table'] = 'diningtable'
clsToPascalCls['tv'] = 'tvmonitor'
clsToPascalCls['motorcycle'] = 'motorbike'
clsToPascalCls['couch'] = 'sofa'; clsToPascalCls['airplane'] = 'aeroplane'
clsToPascalCls['potted plant'] = 'pottedplant'


pascalToCoco = {clsToPascalCls[cls]:cls for cls in clsToPascalCls}
clsToidx = {cls:i for i,cls in enumerate(classes)}
imgToLbl = defaultdict(list)

for cls in tqdm(classes):
    clasToimg = open(clsToPascalCls[cls]+'_trainval.txt').read().splitlines()
    for line in tqdm(clasToimg):
        lsp = line.split()
        if (lsp[0] in trainList) and (int(lsp[1]) == 1):
            imgToLbl[lsp[0]].append(cls)


import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn
import torch.nn.functional as FN
import json
from matplotlib.pyplot import cm
from scipy.stats import binned_statistic

def sigmoid(x):

def plot_binned_stat(allIouVsCls, val_to_plot, bins=10, pltAll = 0, linestyle = ':', plttype = 'stat',applysigx = True , applysigy = False, plt_unitline=False):
    color=cm.rainbow(np.linspace(0,1, len(allIouVsCls.keys())))
    legendK = []
    if pltAll:
        legendK = allIouVsCls.keys()
        for i, cls in enumerate(allIouVsCls):
            xval = FN.sigmoid(torch.FloatTensor(allIouVsCls[cls][val_to_plot[0]])).numpy() if applysigx else allIouVsCls[cls][val_to_plot[0]]
            yval = FN.sigmoid(torch.FloatTensor(allIouVsCls[cls][val_to_plot[1]])).numpy() if applysigy else allIouVsCls[cls][val_to_plot[1]]
            if plttype == 'stat':
                aClsVsRec = binned_statistic(xval, yval,statistic='mean', bins=bins)
                aClsVsRec_std = binned_statistic(xval, yval,statistic=np.std, bins=bins)
                #plt.plot((aClsVsRec[1][:-1]+aClsVsRec[1][1:])/2, aClsVsRec[0],color=color[i],marker='o',linestyle=linestyle);
                plt.errorbar((aClsVsRec[1][:-1]+aClsVsRec[1][1:])/2, aClsVsRec[0], yerr = aClsVsRec_std[0], color=color[i],marker='o',linestyle=linestyle);
            else:
                plt.scatter(xval, yval,alpha=0.5,color=color[i],s=20)
    if pltAll < 2:
        legendK = legendK + ['all']
        allX = np.concatenate([allIouVsCls[cls][val_to_plot[0]] for cls in allIouVsCls])
        allY = np.concatenate([allIouVsCls[cls][val_to_plot[1]] for cls in allIouVsCls])
        xval = FN.sigmoid(torch.FloatTensor(allX)).numpy() if applysigx else allX
        yval = FN.sigmoid(torch.FloatTensor(allY)).numpy() if applysigy else allY
        if plttype == 'stat':
            aClsVsRec = binned_statistic(xval, yval,statistic='mean', bins=bins)
            aClsVsRec_std = binned_statistic(xval, yval,statistic=np.std, bins=bins)
            #plt.plot((aClsVsRec[1][:-1]+aClsVsRec[1][1:])/2, aClsVsRec[0],color=color[-1],marker='o',linestyle='-', linewidth=2);
            plt.errorbar((aClsVsRec[1][:-1]+aClsVsRec[1][1:])/2, aClsVsRec[0], yerr = aClsVsRec_std[0], color=color[-1],marker='o',linestyle='-', linewidth=2);
        else:
            plt.scatter(xval,yval,alpha=0.4,color=color[-1],s=20)
    plt.xlabel(val_to_plot[0])
    plt.ylabel(val_to_plot[1])
    plt.legend(legendK)
    if plt_unitline:
        plt.plot(xval,xval, 'k-');
    plt.show()

fname = 'removeEvalResults/fullres/train_checkpoint_stargan_coco_fulleditor_LowResMask_pascal_RandDiscrWdecay_wgan_30pcUnion_noGT_reg_biasM_randRot_fixedD_randDisc_smM_fixInp_imnet_IN_maxPool_V2_180_1227'
tr_res = json.load(open(fname,'r'))
selected_attrs = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train', 'bottle', 'couch', "dining table", "potted plant", 'chair','tv']


attToIdx = {att:i for i,att in enumerate(selected_attrs)}

res = tr_res
allIouVsCls = {}
for key,img in res['images'].items():
    for cls in img['perclass']:
        if cls not in allIouVsCls:
            allIouVsCls[cls] = {'iou':[], 'recall':[], 'precision':[], 'ocls':[],'acls':[],'gtsize':[], 'predsize':[], 'false_damage':[], 'n_obj':[], 'diff':[]}
        allIouVsCls[cls]['iou'].append(img['perclass'][cls]['iou'])
        allIouVsCls[cls]['recall'].append(img['perclass'][cls]['rec'])
        allIouVsCls[cls]['precision'].append(img['perclass'][cls]['prec'])
        allIouVsCls[cls]['ocls'].append(img['real_scores'][attToIdx[cls]])
        allIouVsCls[cls]['acls'].append(img['perclass'][cls]['remove_scores'][attToIdx[cls]])
        allIouVsCls[cls]['rSucc'].append(float(img['perclass'][cls]['remove_scores'][attToIdx[cls]]<0.))
        allIouVsCls[cls]['diff'].append(img['real_scores'][attToIdx[cls]] - img['perclass'][cls]['remove_scores'][attToIdx[cls]])
        allIouVsCls[cls]['gtsize'].append(img['perclass'][cls]['gtSize'])
        allIouVsCls[cls]['predsize'].append(img['perclass'][cls]['predSize'])
        #allIouVsCls[cls]['false_damage'].append(np.max([img['real_scores'][oclsId] - img['perclass'][cls]['remove_scores'][oclsId] for oclsId in img['real_label']  if selected_attrs[oclsId]!=cls])/(len(img['real_label'])-1+1e-6) )
        allIouVsCls[cls]['false_damage'].append(np.max([img['real_scores'][oclsId] - img['perclass'][cls]['remove_scores'][oclsId] for oclsId in img['real_label'] if selected_attrs[oclsId]!=cls]) if len(img['real_label'])>1 else np.nan)
        allIouVsCls[cls]['n_obj'].append(len(img['real_label']))


val_to_plot = ['ocls','recall']





'person' ,'bird' , 'cat' , 'cow' , 'dog' ,  'horse' ,  'sheep' , 'airplane' , 'bicycle' ,'boat' , 'bus' , 'car' , 'motorcycle' ,  'train' , 'bottle' ,  'couch' , 'dining table' , 'potted plant',  'chair' ,  'tv'


cat2id= {}
data = {} ; data['images'] = {}
for ann in train_ann:
    annSp = ann.split()
    imgid = int(annSp[0].split('.')[0])
    cls = annSp[1].lower()
    if imgid not in data['images']:
        finfo = subprocess.check_output(['file', 'flickr_logos_27_dataset_images/'+annSp[0]])
        data['images'][imgid] = {'bboxAnn': [], 'id': imgid, 'filename':annSp[0], 'split':'train','imgSize': map(int, finfo.split(',')[-2].split('x'))}
    if cls not in cat2id:
        cat2id[cls] = len(cat2id)

    bbox = map(int,annSp[-4:])
    img_w,img_h = data['images'][imgid]['imgSize']
    bbox = [float(bbox[0])/float(img_w), float(bbox[1])/float(img_h), float(bbox[2]-bbox[0])/float(img_w), float(bbox[3] - bbox[1])/float(img_h)]
    data['images'][imgid]['bboxAnn'].append({'bbox': bbox, 'cid': cat2id[cls]})

data['categories'] = [{'id':cat2id[cat], 'name':cat} for cat in cat2id]



for ann in val_ann:
    annSp = ann.split()
    imgid = int(annSp[0].split('.')[0])
    cls = annSp[1].lower()
    if imgid not in data['images']:
        finfo = subprocess.check_output(['file', 'flickr_logos_27_dataset_images/'+annSp[0]])
        data['images'][imgid] = {'bboxAnn': [], 'id': imgid, 'filename':annSp[0], 'split':'train','imgSize': map(int, finfo.split(',')[-2].split('x'))}
    if cls not in cat2id:
        cat2id[cls] = len(cat2id)

    bbox = [0., 0., 1., 1.]
    data['images'][imgid]['bboxAnn'].append({'bbox': bbox, 'cid': cat2id[cls]})


for ann in val_ann:
    annSp = ann.split()
    imgid = int(annSp[0].split('.')[0])
    cls = annSp[1].lower()
    data['images'][imid2index[imgid]]['split'] = 'val'



cat2id= {}
data = {} ; data['images'] = {}
for ann in tqdm(train_ann):
    annSp = ann.split()
    if annSp[4]:
        imgid = int(annSp[2].split('.')[0])
        cls = annSp[1].lower()
        if imgid not in data['images']:
            finfo = subprocess.check_output(['file', 'images/'+annSp[2]])
            data['images'][imgid] = {'bboxAnn': [], 'id': imgid, 'filename':annSp[2], 'split':'train','imgSize': map(int, finfo.split(',')[-2].split('x'))}
        if cls not in cat2id:
            cat2id[cls] = len(cat2id)

        bbox = map(int,annSp[-4:])
        img_w,img_h = data['images'][imgid]['imgSize']
        bbox = [float(bbox[0])/float(img_w), float(bbox[1])/float(img_h), float(bbox[2]-bbox[0])/float(img_w), float(bbox[3] - bbox[1])/float(img_h)]
        data['images'][imgid]['bboxAnn'].append({'bbox': bbox, 'cid': cat2id[cls]})

data['categories'] = [{'id':cat2id[cat], 'name':cat} for cat in cat2id]

for fname in tqdm(notPresentImgs):
    finfo = subprocess.check_output(['file', 'images/'+fname])
    imgid = int(fname.split('.')[0])
    data['images'].append({'bboxAnn': [], 'id': imgid, 'filename':fname, 'split':'train','imgSize': map(int, finfo.split(',')[-2].split('x'))})



import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import seaborn
from PIL import Image
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

fig, ax = plt.subplots();
ax.imshow(img, origin='upper', extent=[0,128, 128,0]);
axins = zoomed_inset_axes(ax, zoom=3, loc=7)
extent = [50, 60, 70, 60]
axins.imshow(img, interpolation="nearest", origin='upper', extent=[0,128, 0,128])
axins.set_xlim(*extent[:2])
axins.set_ylim(*extent[2:])
axins.yaxis.get_major_locator().set_params(nbins=7)
axins.xaxis.get_major_locator().set_params(nbins=7)
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
ax.set_axis_off()
plt.draw(); plt.show()


fig, ax = plt.subplots(frameon=False);
ax.imshow(img, origin='lower');
axins = zoomed_inset_axes(ax, zoom=3, loc=7)
extent = [55, 65, 44, 54]
axins.imshow(img, interpolation="nearest", origin='lower')
axins.set_xlim(*extent[:2])
axins.set_ylim(*extent[2:])
axins.yaxis.get_major_locator().set_params(nbins=7)
axins.xaxis.get_major_locator().set_params(nbins=7)
#axins.set_axis_off()
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="r")
ax.set_axis_off()
plt.draw(); plt.show()




import numpy as np
import json
from scipy.special import expit
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.1f'
    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Removed Object')
    plt.xlabel('Change in Classifier Scores after removal')


allRes = json.load(open('testRobustModel/val_checkpoint_stargan_coco_fulleditor_LowResMask_pascal_RandDiscrWdecay_wgan_30pcUnion_noGT_imnet_V2_msz32_ftuneMask_withPmask_L1150_tv_nb4_styleloss3k_248_1570_withRobustClassifier','r'))
selected_attrs = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train', 'bottle', 'couch', "dining table", "potted plant", 'chair', 'tv']




allCoOccur = np.zeros((2302,20,20))
allClassifierScores = np.zeros((2302,20))
allClassifierScoresEditNormChange = np.zeros((2302,20,20))
allClassifierScoresEdit = np.zeros((2302,20,20))
clsToids = {cls:i for i,cls in enumerate(selected_attrs)}
allLabels = np.zeros((2302,20))
for i,k in enumerate(allRes['images'].keys()):
    allClassifierScores[i,:] = allRes['images'][k]['real_scores']
    allLabels[i,allRes['images'][k]['real_label']] = 1
    for cls in allRes['images'][k]['perclass']:
        allCoOccur[i,clsToids[cls],allRes['images'][k]['real_label']] = 1
        allClassifierScoresEdit[i,clsToids[cls],allRes['images'][k]['real_label']] = np.array(allRes['images'][k]['perclass'][cls]['remove_scores'])[allRes['images'][k]['real_label']] - allClassifierScores[i,allRes['images'][k]['real_label']]
        allClassifierScoresEditNormChange[i,clsToids[cls],allRes['images'][k]['real_label']] = expit(np.array(allRes['images'][k]['perclass'][cls]['remove_scores'])[allRes['images'][k]['real_label']]) - expit(allClassifierScores[i,allRes['images'][k]['real_label']])



n_class = len(selected_attrs)

perclass_tp = np.zeros((2,len(selected_attrs)))
perclass_fn = np.zeros((2,len(selected_attrs)))
perclass_fp = np.zeros((2,len(selected_attrs)))

for k,img in allRes['images'].items():
    nonexistLabel = [i for i in xrange(len(selected_attrs)) if i not in img['real_label']]
    if (0 not in img['real_label']):
        perclass_tp[0,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]>0)
        perclass_fn[0,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]<=0)
        perclass_fp[0, nonexistLabel] += (np.array(img['real_scores'])[nonexistLabel]>0)
    else:
        perclass_tp[1,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]>0)
        perclass_fn[1,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]<=0)
        perclass_fp[1, nonexistLabel] += (np.array(img['real_scores'])[nonexistLabel]>0)

perclass_tp_nr = np.zeros((2,len(selected_attrs)))
perclass_fn_nr = np.zeros((2,len(selected_attrs)))
perclass_fp_nr = np.zeros((2,len(selected_attrs)))

for k,img in allResNonRob['images'].items():
    nonexistLabel = [i for i in xrange(len(selected_attrs)) if i not in img['real_label']]
    if (0 not in img['real_label']):
        perclass_tp_nr[0,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]>0)
        perclass_fn_nr[0,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]<=0)
        perclass_fp_nr[0, nonexistLabel] += (np.array(img['real_scores'])[nonexistLabel]>0)
    else:
        perclass_tp_nr[1,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]>0)
        perclass_fp_nr[1,nonexistLabel] += (np.array(img['real_scores'])[nonexistLabel]>0)
        perclass_fn_nr[1,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]<=0)


recall = perclass_tp/(perclass_tp+perclass_fn+1e-6)
precision = perclass_tp/(perclass_tp+perclass_fp+1e-6)
f1_score = 2.0* (recall*precision)/(recall+precision+1e-6)
recall_nr = perclass_tp_nr/(perclass_tp_nr+perclass_fn_nr+1e-6)
precision_nr = perclass_tp_nr/(perclass_tp_nr+perclass_fp_nr+1e-6)
f1_score_nr = 2.0* (recall_nr*precision_nr)/(recall_nr+precision_nr+1e-6)



allMf1s = []
allTh = []
alLabels = np.zeros((len(allRes['images']),n_class))
allPred = np.zeros((len(allRes['images']),n_class))
for i,k in enumerate(allRes['images']):
    alLabels[i,allRes['images'][k]['real_label']] = 1
    allPred[i,:] = allRes['images'][k]['real_scores']

for i in xrange(len(selected_attrs)):
    pr,rec,th = precision_recall_curve(alLabels[:,i],allPred[:,i]);
    f1s = 2*(pr*rec)/(pr+rec+1e-6); mf1idx = np.argmax(f1s);
    print 'Max f1 = %.2f, th =%.2f'%(f1s[mf1idx], th[mf1idx]);
    allMf1s.append(f1s[mf1idx])
    allTh.append(th[mf1idx])

allMf1s_nr = []
allTh_nr = []
alLabels_nr = np.zeros((len(allRes['images']),n_class))
allPred_nr = np.zeros((len(allRes['images']),n_class))
for i,k in enumerate(allResNonRob['images']):
    alLabels_nr[i,allResNonRob['images'][k]['real_label']] = 1
    allPred_nr[i,:] = allResNonRob['images'][k]['real_scores']
for i in xrange(len(selected_attrs)):
    pr,rec,th = precision_recall_curve(alLabels[:,i],allPred[:,i]);
    f1s = 2*(pr*rec)/(pr+rec+1e-6); mf1idx = np.argmax(f1s);
    print 'Max f1 = %.2f, th =%.2f'%(f1s[mf1idx], th[mf1idx]);
    allMf1s_nr.append(f1s[mf1idx])
    allTh_nr.append(th[mf1idx])
allTh = np.array(allTh)
allTh_nr = np.array(allTh_nr)

perclass_tp = np.zeros((2,len(selected_attrs)))
perclass_fn = np.zeros((2,len(selected_attrs)))
perclass_fp = np.zeros((2,len(selected_attrs)))

for k,img in allRes['images'].items():
    nonexistLabel = [i for i in xrange(len(selected_attrs)) if i not in img['real_label']]
    if (0 not in img['real_label']):
        perclass_tp[0,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]>allTh[img['real_label']])
        perclass_fn[0,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]<=allTh[img['real_label']])
        perclass_fp[0, nonexistLabel] += (np.array(img['real_scores'])[nonexistLabel]>allTh[nonexistLabel])
    else:
        perclass_tp[1,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]>allTh[img['real_label']])
        perclass_fn[1,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]<=allTh[img['real_label']])
        perclass_fp[1, nonexistLabel] += (np.array(img['real_scores'])[nonexistLabel]>allTh[nonexistLabel])

perclass_tp_nr = np.zeros((2,len(selected_attrs)))
perclass_fn_nr = np.zeros((2,len(selected_attrs)))
perclass_fp_nr = np.zeros((2,len(selected_attrs)))

for k,img in allResNonRob['images'].items():
    nonexistLabel = [i for i in xrange(len(selected_attrs)) if i not in img['real_label']]
    if (0 not in img['real_label']):
        perclass_tp_nr[0,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]>allTh_nr[img['real_label']])
        perclass_fn_nr[0,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]<=allTh_nr[img['real_label']])
        perclass_fp_nr[0, nonexistLabel] += (np.array(img['real_scores'])[nonexistLabel]>allTh_nr[nonexistLabel])
    else:
        perclass_tp_nr[1,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]>allTh_nr[img['real_label']])
        perclass_fp_nr[1,nonexistLabel] += (np.array(img['real_scores'])[nonexistLabel]>allTh_nr[nonexistLabel])
        perclass_fn_nr[1,img['real_label']] += (np.array(img['real_scores'])[img['real_label']]<=allTh_nr[img['real_label']])


recall = perclass_tp/(perclass_tp+perclass_fn+1e-6)
precision = perclass_tp/(perclass_tp+perclass_fp+1e-6)
f1_score = 2.0* (recall*precision)/(recall+precision+1e-6)
recall_nr = perclass_tp_nr/(perclass_tp_nr+perclass_fn_nr+1e-6)
precision_nr = perclass_tp_nr/(perclass_tp_nr+perclass_fp_nr+1e-6)
f1_score_nr = 2.0* (recall_nr*precision_nr)/(recall_nr+precision_nr+1e-6)

recall_ovr = perclass_tp.sum()/(perclass_tp.sum()+perclass_fn.sum()+1e-6)
precision_ovr = perclass_tp.sum()/(perclass_tp.sum()+perclass_fp.sum()+1e-6)
recall_ovr_nr = perclass_tp_nr.sum()/(perclass_tp_nr.sum()+perclass_fn_nr.sum()+1e-6)
precision_ovr_nr = perclass_tp_nr.sum()/(perclass_tp_nr.sum()+perclass_fp_nr.sum()+1e-6)
f1_score_ovr = 2.0* (recall_ovr*precision_ovr)/(recall_ovr+precision_ovr+1e-6)
f1_score_ovr_nr = 2.0* (recall_ovr_nr*precision_ovr_nr)/(recall_ovr_nr+precision_ovr_nr+1e-6)

recall_ovr2 = perclass_tp.sum(axis=1)/(perclass_tp.sum(axis=1)+perclass_fn.sum(axis=1)+1e-6)
precision_ovr2 = perclass_tp.sum(axis=1)/(perclass_tp.sum(axis=1)+perclass_fp.sum(axis=1)+1e-6)
recall_ovr2_nr = perclass_tp_nr.sum(axis=1)/(perclass_tp_nr.sum(axis=1)+perclass_fn_nr.sum(axis=1)+1e-6)
precision_ovr2_nr = perclass_tp_nr.sum(axis=1)/(perclass_tp_nr.sum(axis=1)+perclass_fp_nr.sum(axis=1)+1e-6)
f1_score_ovr2 = 2.0* (recall_ovr2*precision_ovr2)/(recall_ovr2+precision_ovr2+1e-6)
f1_score_ovr2_nr = 2.0* (recall_ovr2_nr*precision_ovr2_nr)/(recall_ovr2_nr+precision_ovr2_nr+1e-6)

print f1_score_ovr2
print f1_score_ovr2_nr
print precision_ovr2
print precision_ovr2_nr
print recall_ovr2
print recall_ovr2_nr

print('Score : || %s |'%(' | '.join(['%6s'%att[:6] for att in selected_attrs])))
print('F1woPR: || %s |'%(' | '.join(['  %.2f' % sc for sc in f1_score[0,:]])))
print('F1woP : || %s |'%(' | '.join(['  %.2f' % sc for sc in f1_score_nr[0,:]])))
print('\nRwoPR : || %s |'%(' | '.join(['  %.2f' % sc for sc in recall[0,:]])))
print('RwoP  : || %s |'%(' | '.join(['  %.2f' % sc for sc in recall_nr[0,:]])))
print('\nPwoPR : || %s |'%(' | '.join(['  %.2f' % sc for sc in precision[0,:]])))
print('PwoP  : || %s |'%(' | '.join(['  %.2f' % sc for sc in precision_nr[0,:]])))

print('\n\nScore : || %s |'%(' | '.join(['%6s'%att[:6] for att in selected_attrs])))
print('F1wPR : || %s |'%(' | '.join(['  %.2f' % sc for sc in f1_score[1,:]])))
print('F1wP  : || %s |'%(' | '.join(['  %.2f' % sc for sc in f1_score_nr[1,:]])))
print('\nRwPR  : || %s |'%(' | '.join(['  %.2f' % sc for sc in recall[1,:]])))
print('RwP   : || %s |'%(' | '.join(['  %.2f' % sc for sc in recall_nr[1,:]])))
print('\nPwPR  : || %s |'%(' | '.join(['  %.2f' % sc for sc in precision[1,:]])))
print('PwP   : || %s |'%(' | '.join(['  %.2f' % sc for sc in precision_nr[1,:]])))



selected_attrs = ['person' , 'bicycle' , 'car' , 'motorcycle' , 'airplane' , 'bus' , 'train' , 'truck' , 'boat' , 'traffic light' , 'fire hydrant' , 'stop sign' , 'parking meter' , 'bench' , 'bird' , 'cat' , 'dog' , 'horse' , 'sheep' , 'cow' , 'elephant' , 'bear' , 'zebra' , 'giraffe' , 'backpack' , 'umbrella' , 'handbag' , 'tie' , 'suitcase' , 'frisbee' , 'skis' , 'snowboard' , 'sports ball' , 'kite' , 'baseball bat' , 'baseball glove' , 'skateboard' , 'surfboard' , 'tennis racket' , 'bottle' , 'wine glass' , 'cup' , 'fork' , 'knife' , 'spoon' , 'bowl' , 'banana' , 'apple' , 'sandwich' , 'orange' , 'broccoli' , 'carrot' , 'hot dog' , 'pizza' , 'donut' , 'cake' , 'chair' , 'couch' , 'potted plant' , 'bed' , 'dining table' , 'toilet' , 'tv' , 'laptop' , 'mouse' , 'remote' , 'keyboard' , 'cell phone' , 'microwave' , 'oven' , 'toaster' , 'sink' , 'refrigerator' , 'book' , 'clock' , 'vase' , 'scissors' , 'teddy bear' , 'hair drier' , 'toothbrush']



for i in xrange(len(selected_attrs)):
    pr,rec,th = precision_recall_curve(alLabels[:,i],allPred[:,i]);
    plt.plot(pr,rec,label=selected_attrs[i]+'_r')

for i in xrange(len(selected_attrs)):
     pr,rec,th = precision_recall_curve(alLabels_nr[:,i],allPred_nr[:,i]);
     plt.plot(pr,rec,label=selected_attrs[i]+'_nr',linestyle=':')
plt.legend(ncol=2); plt.show()

imids = resRob.keys()

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
    return ap, mrec, mprec


def computeAP(allSc, allLb):
    si = (-allSc).argsort(axis=0)
    cid = np.arange(allLb.shape[1])
    tp = allLb[si[:,cid],cid] > 0.
    fp = allLb[si[:,cid],cid] == 0.
    tp = tp.cumsum(axis=0).astype(np.float32)
    fp = fp.cumsum(axis=0).astype(np.float32)
    rec = (tp+1e-8)/((allLb>0.)+1e-8).sum(axis=0).astype(np.float32)
    prec = (tp+1e-8)/ (tp+ fp+1e-8)
    ap,mrec,mprec = VOCap(rec,prec)

    return ap,mrec,mprec




import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
import json

n_class = len(selected_attrs)

alLabels = np.zeros((len(allRes['images']),n_class))
allPred = np.zeros((len(allRes['images']),n_class))
for i,k in enumerate(allRes['images']):
    alLabels[i,allRes['images'][k]['real_label']] = 1
    allPred[i,:] = allRes['images'][k]['real_scores']

alLabels_nr = np.zeros((len(allRes['images']),n_class))
allPred_nr = np.zeros((len(allRes['images']),n_class))
for i,k in enumerate(allRes['images']):
    alLabels_nr[i,allResNonRob['images'][k]['real_label']] = 1
    allPred_nr[i,:] = allResNonRob['images'][k]['real_scores']

ap_r, mrec_r, mprec_r = computeAP(allPred,alLabels)
ap_nr, mrec_nr, mprec_nr = computeAP(allPred_nr,alLabels)

n_high= 15
print ap_r.mean(), ap_nr.mean()
mdiff = np.argsort(np.abs(ap_r - ap_nr))[::-1]
print('Score: || %s |'%(' | '.join(['%6s'%selected_attrs[mdiff[i]][:6] for i in xrange(n_high)])))
print('AP_R : || %s |'%(' | '.join(['  %.2f' % ap_r[mdiff[i]] for i in xrange(n_high)])))
print('AP_NR: || %s |'%(' | '.join(['  %.2f' % ap_nr[mdiff[i]] for i in xrange(n_high)])))

color=cm.Spectral(np.linspace(0,1,n_high))
for i in xrange(n_high):
    #pr,rec,th = precision_recall_curve(alLabels[:,mdiff[i]],allPred[:,mdiff[i]]);
    plt.plot(mrec_r[:,mdiff[i]],mprec_r[:,mdiff[i]],label=selected_attrs[mdiff[i]]+'_r',c=color[i])

for i in xrange(n_high):
     #pr,rec,th = precision_recall_curve(alLabels_nr[:,mdiff[i]],allPred_nr[:,mdiff[i]]);
     plt.plot(mrec_nr[:,mdiff[i]],mprec_nr[:,mdiff[i]],label=selected_attrs[mdiff[i]]+'_nr',linestyle=':',c=color[i])
plt.legend(ncol=2);
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Top %d classes with largest difference in ap'%(n_high))
plt.show()

isp = (alLabels[:,0] == 1.)
ap_r_wp, mrec_r_wp, mprec_r_wp = computeAP(allPred[isp,1:],alLabels[isp,1:])
ap_r_wop, mrec_r_wop, mprec_r_wop = computeAP(allPred[~isp,1:],alLabels[~isp,1:])

ap_nr_wp, mrec_nr_wp, mprec_nr_wp = computeAP(allPred_nr[isp,1:],alLabels_nr[isp,1:])
ap_nr_wop, mrec_nr_wop, mprec_nr_wop = computeAP(allPred_nr[~isp,1:],alLabels_nr[~isp,1:])

print ap_r_wp.mean(), ap_r_wop.mean()
print ap_nr_wp.mean(), ap_nr_wop.mean()
mdiff = np.argsort(np.abs(ap_r_wp - ap_nr_wp))[::-1]
print('Score: || %s |'%(' | '.join(['%6s'%selected_attrs[mdiff[i]][:6] for i in xrange(n_high)])))
print('AP_R : || %s |'%(' | '.join(['  %.2f' % ap_r_wp[mdiff[i]] for i in xrange(n_high)])))
print('AP_NR: || %s |'%(' | '.join(['  %.2f' % ap_nr_wp[mdiff[i]] for i in xrange(n_high)])))

color=cm.Spectral(np.linspace(0,1,n_high))
for i in xrange(n_high):
    #pr,rec,th = precision_recall_curve(alLabels[isp,mdiff[i]],allPred[isp,mdiff[i]]);
    plt.plot(mrec_r_wp[:,mdiff[i]],mprec_r_wp[:,mdiff[i]],label=selected_attrs[mdiff[i]]+'_r',c=color[i])

for i in xrange(n_high):
     pr,rec,th = precision_recall_curve(alLabels_nr[isp,mdiff[i]],allPred_nr[isp,mdiff[i]]);
     plt.plot(mrec_nr_wp[:,mdiff[i]],mprec_nr_wp[:,mdiff[i]],label=selected_attrs[mdiff[i]]+'_nr',linestyle=':',c=color[i])
plt.legend(ncol=2);
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Top %d classes with largest difference in ap'%(n_high))
plt.show()



#================================================================================================
#=============================   Co ouccerence computations =====================================
#================================================================================================
def plot_cooccur_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, vmin=None, vmax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.diagonal()[:, np.newaxis]
        print("Normalized co-occurance matrix (w.r.t primary class counts)")
    else:
        print('Co-occurance matrix, without normalization')

    print(cm)

    if vmin is None:
        vmin = cm.min()
    if vmax is None:
        vmin = cm.max()

    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.1f'
    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Primary Class')
    plt.xlabel('Co-occuring class')


cidToCls = {cat['id']:cat['name'] for cat in data['categories']}
clsToids = {cls:i for i,cls in enumerate(selected_attrs)}
allCoOccur = np.zeros((80,80))
for img in data['images']:
    if img['split'] == 'train':
        imgCls = set()
        for bb in img['bboxAnn']:
            imgCls.add(cidToCls[bb['cid']])
        for clsX in list(imgCls):
            for clsY in list(imgCls):
                allCoOccur[clsToids[clsX],clsToids[clsY]] +=1


plot_cooccur_matrix(np.log(allCoOccur+1e-1),selected_attrs,normalize=False,cmap=plt.cm.plasma); plt.show()



indivOccurenceNp = np.zeros(80)
clsCounts = np.zeros(80)
for img in data['images']:
    if img['split'] == 'train':
        imgCls = set()
        for bb in img['bboxAnn']:
            imgCls.add(cidToCls[bb['cid']])
        imgCls = list(imgCls)
        if len(imgCls)==1:
            indivOccurenceNp[clsToids[imgCls[0]]] += 1
        else:
            clsCounts[[clsToids[cls] for cls in imgCls]] += 1


occurStats = np.log10(allCoOccur.diagonal())
fig, ax = plt.subplots()
ax.scatter(occurStats,aps)
for i, txt in enumerate(selected_attrs):
    ax.annotate(txt, (occurStats[i],aps[i]))



fig, ax = plt.subplots()
texts = []
for i, txt in enumerate(selected_attrs):
    texts.append(ax.annotate(txt, (aps[i]+0.003,rvox[i]+0.5),fontsize=14))
ax.scatter(aps,rvox,s=50,alpha=1.0)
plt.xlabel('Average precision for the class', fontsize=14)
plt.ylabel('% Violations to changes in context', fontsize=14)

plt.show()


fig, ax = plt.subplots()
ax.plot(np.arange(0.0,rmax_reg.max(),0.05),np.arange(0.0, rmax_reg.max(),0.05),'k-');
sax = ax.scatter(rmax_reg, rmax_da, c=apReg, s=24,cmap=plt.cm.plasma);
cbar = fig.colorbar(sax,ax=ax)
cbar.set_label('Average precision of the class', fontsize=16)
ax.set_xlabel('% violations in the original model', fontsize=16)
ax.set_ylabel('% violations in the Data augmented model',fontsize=16)
texts = []
th = 0.1
for i, txt in enumerate(selected_attrs):
    if rmax_reg[i] - rmax_da[i] > th:
        texts.append(ax.annotate(txt, (rmax_reg[i]-0.03,rmax_da[i]-0.03),fontsize=14))
    elif rmax_reg[i] - rmax_da[i] < -th:
        texts.append(ax.annotate(txt, (rmax_reg[i]+0.03,rmax_da[i]+0.03),fontsize=14))

plt.show()
