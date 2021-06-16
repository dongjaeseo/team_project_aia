import sys
sys.path.append('C:\\tp\\mask')

import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG', 'lane']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
model.load_weights(filepath="C:\\tp\\lane_mask_rcnn_trained.h5", 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
image = cv2.imread("C:\\Users\\ai\\Desktop\\test.jpg")
# image = cv2.imread("C:\\Users\\ai\\Desktop\\20210529_152707.jpg")
# image = cv2.imread("C:\\Users\\ai\\Desktop\\20210529_152707(1).jpg")
# image = cv2.imread("C:\\tp\\mask\\lane_detect\\lane\\images\\20210529_152438.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)

# Get the results for the first image.
r = r[0]

# Visualize the detected objects.
# mrcnn.visualize.display_instances(image=image,
#                                   boxes=r['rois'],
#                                   masks=r['masks'],
#                                   class_ids=r['class_ids'],
#                                   class_names=CLASS_NAMES,
                                #   scores=r['scores'])

import numpy as np
import matplotlib.pyplot as plt
# plt.imshow(r['masks'][:,:,0])
# plt.show()

# print(r['masks'].shape) # (4032, 3024, 1)
# print(np.where(r['masks']==True))
# print(r['masks'][1344].shape)
a = np.where(r['masks'][1344]==True)[0]
print(a)
print((a[-1]+a[0])/2)

import cv2
import time
cap = cv2.VideoCapture('C:/tp/video/vid0.mov')
frame_rate = 10
prev = 0

while True: # 무한 루프
    time_elapsed = time.time() - prev
    ret, frame = cap.read() # 두 개의 값을 반환하므로 두 변수 지정

    if not ret: # 새로운 프레임을 못받아 왔을 때 braek
        break

    if time_elapsed > 1./frame_rate:
        prev = time.time()

    if ret:
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow('img', img)
        cv2.waitKey(0)

