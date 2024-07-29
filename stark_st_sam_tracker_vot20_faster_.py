#!/usr/bin/python

# This is a simple example of a tracker implemented in Python. It uses OpenCV to compute the normalized cross correlation and find the best match with a template from a first frame.
# The demo is implemented using VOT Manager to show how a single object tracker can be quickly adapted to multi object scenarios.

import vot
import cv2
import glob
import numpy as np
import sys
import os
prj_path = '/usr/mvl2/esdft/Stark/'

sys.path.append(prj_path)

from lib.test.evaluation import Tracker
from lib.test.evaluation.environment import env_settings
from vot_data_preprocessing import get_bbox



from segment_anything import sam_model_registry, SamPredictor

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

sam_checkpoint = "/usr/mvl2/esdft/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)



def _build_init_info(box):
    return {'init_bbox': box}




class mixformerTracker(object):

    def __init__(self, tracker_name='stark_st', tracker_param='baseline'):

        #tracker_params = {'model': 'mixformer_vit_base_online.pth.tar', 'update_interval': 10, 'online_sizes': 5, 'search_area_scale': 4.5, 'max_score_decay': 1.0, 'vis_attn': 0}
        #tracker_params = {'model': 'mixformer_vit_base_online.pth.tar'}
        tracker_info = Tracker(tracker_name, tracker_param, "vot20", None)
        params = tracker_info.get_parameters()
        #params.tracker_name = tracker_name
        #params.tracker_param  = tracker_param
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)
        #self.H, self.W, _ = img_rgb.shape
        #self.tracker.initialize(img_rgb, _build_init_info(bbox))


    def initialize(self, img_rgb, bbox):

        self.H, self.W, _ = img_rgb.shape
        self.tracker.initialize(img_rgb, _build_init_info(bbox))


    '''def track(self, img_rgb):


        #img_rgb = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        out = self.tracker.track(img_rgb)
        #predictor.set_image(img_rgb)
        box = out['target_bbox']
        #input_box = np.array([box[0], box[1], box[0]+box[2], box[1]+box[3]])
        #masks, _, _ = predictor.predict(point_coords=None,point_labels=None,box=input_box[None, :],multimask_output=False,)

        #temp = masks[0]*1
        #mask = temp.astype(np.uint8)

        return box'''





from vot_data_preprocessing import _mask_to_bbox



handle = vot.VOT("mask", multiobject=True)
imagefile = handle.frame()


objects = handle.objects()
#image = cv2.imread(imagefile)
image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)


trackers = [mixformerTracker() for object in objects]


for ind in range(len(trackers)):
    trackers[ind].initialize(image, _mask_to_bbox(objects[ind]))


while True:

    imagefile = handle.frame()
    if not imagefile:
        break

    #image = cv2.imread(imagefile)
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    #handle.report([tracker.track(image) for tracker in trackers])

    pred_list = []


    for tracker in trackers:

        box = tracker.tracker.track(image)['target_bbox']
        input_box = np.array([box[0], box[1], box[0]+box[2], box[1]+box[3]])
        masks, scores, _ = predictor.predict(point_coords=None,point_labels=None,box=input_box[None, :],multimask_output=True,)
        #temp = masks[0]*1
        temp = masks[np.argmax(scores)]
        mask = temp.astype(np.uint8)
        pred_list.append(mask)



    handle.report(pred_list)


'''images_path = '/usr/mvl2/esdft/cat/' 
frames = glob.glob(f'{images_path}/*.jpg') 
file_path = '/usr/mvl2/esdft/development_data/sequences/cat-18/'

gt_files = glob.glob(f'{file_path}/groundtruth*.txt') 
bboxes = [] 


for gt_file in gt_files:
    bboxes.append(get_bbox(gt_file))



imagefile = frames[0]
image = cv2.imread(imagefile)


trackers = [mixformerTracker(img_rgb=image, bbox=bbox) for bbox in bboxes]


color = np.array([200, 200, 250])


for frame in frames:

    image = cv2.imread(frame)

    predictor.set_image(image)

 
    for tracker in trackers:
        mask = tracker.track(image)

        #h, w = mask.shape[-2:]
        #mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        #cv2.imshow('mask', mask_image)

        image[mask==1] = color


        #cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), color=(255, 0, 0), thickness=5)


    cv2.imshow('image', image) 
    cv2.waitKey(1)'''
