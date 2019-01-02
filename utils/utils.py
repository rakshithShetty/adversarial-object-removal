from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geo
from shapely.ops import cascaded_union

def intersectBox(bboxA, bboxB):

    # Co-ordinates are of the form, [topleft X, topleft Y, width, height]
    topLeft = [max(bboxA[0],bboxB[0]), max(bboxA[1], bboxB[1])]
    botRight = [min(bboxA[0]+bboxA[2],bboxB[0]+bboxB[2]), min(bboxA[1]+bboxA[3], bboxB[1]+bboxB[3])]
    inter = max(botRight[0] - topLeft[0],0) * max(botRight[1] - topLeft[1],0)

    return inter

#def intersectNP(bboxA, bboxB):
#    # Co-ordinates are of the form, [topleft X, topleft Y, width, height]
#    topLeft = np.maximum(bboxA[:2],bboxB[:2])
#    botRight = np.minimum(bboxA[:2]+bboxA[2:],bboxB[:2]+bboxB[2:])
#    inter = np.prod(np.clip(botRight-topLeft,0, None))
#    return inter


#Compute How much of boxB is in BoxA
def computeContainment(bboxA, bboxB):
    inter = intersectBox(bboxA, bboxB)
    aA = bboxA[2] * bboxA[3]
    aB = bboxB[2] * bboxB[3]
    return (inter/aB), inter, aA, aB

def computeIOU(bboxA, bboxB):
    inter = intersectBox(bboxA, bboxB)
    aA = bboxA[2] * bboxA[3]
    aB = bboxB[2] * bboxB[3]
    aU = aA + aB - inter
    return (inter/aU), (inter/aA), (inter/aB)

def computeUnionArea(boxes):
    boxes = [geo.box(bb[0],bb[1],bb[0]+bb[2], bb[1]+bb[3]) for bb in boxes]
    return cascaded_union(boxes).area

def show(img):
    npimg = img.numpy()
    plt.imshow(((np.transpose(npimg, (1,2,0))+1.0)*255./2.0).astype(np.uint8), interpolation='nearest')
    plt.show()
