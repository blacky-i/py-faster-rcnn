#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import os
os.environ['GLOG_minloglevel'] = '2' 

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os, sys, cv2
import argparse

import subprocess
import re
import getopt
import json
import datetime
import uuid
from detect_color import detect_colors
import caffe

CLASSES = ('__background__',
           'coat','shirt','tie','collar')

NETS = { 'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                  'VGG_CNN_M_1024_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = image_name
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    #print ('Detection took {:.3f}s for '
    #       '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3
    output={'file':im_file,'date_published':str(datetime.datetime.now())}
    output['classes']=[]
    output['propabilities']=[]
    bboxes=[]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
	inds = np.where(dets[:,-1]>=CONF_THRESH)[0]


	for i in inds:
	    #print(cls,dets[i,-1],dets[i,0],dets[i,1],dets[i,2],dets[i,3])
	    output['classes'].append(cls)
	    output['propabilities'].append(float(dets[i,-1]))
	    bbox={}
	    bbox['x_left']=int(dets[i,0])
	    bbox['y_left']=int(dets[i,1])
	    bbox['x_right']=int(dets[i,2])
	    bbox['y_right']=int(dets[i,3])
	    bboxes.append(bbox)
	output['bboxes']=bboxes

    output['colors']=detect_colors(output)
    #print(json.dumps(output, ensure_ascii=False))
    filename=str(uuid.uuid4().hex)+'.json'
    file=open(filename,'w')
    file.write(json.dumps(output, ensure_ascii=False))
    file.close()
    return file.name
	 #vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg_cnn_m_1024]',
                        choices=NETS.keys(), default='vgg_cnn_m_1024')
    parser.add_argument('--filepath', dest='img_filepath',help='filepath to image')
    parser.add_argument('-f', dest='img_filepath',help='filepath to image')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    #print(cfg.MODELS_DIR,cfg.DATA_DIR)
    prototxt = os.path.join('/home/user/py-faster-rcnn/models/pascal_voc', NETS[args.demo_net][0],
                            'faster_rcnn_end2end_fashion', 'test.prototxt')
    caffemodel = os.path.join('/home/user/py-faster-rcnn/saved_caffe_weights_fashion',
			      NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    #print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    print demo(net,args.img_filepath)
   
