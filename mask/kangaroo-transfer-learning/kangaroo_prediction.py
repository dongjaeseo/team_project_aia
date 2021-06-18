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

for k in range(5):
    cap = cv2.VideoCapture(f'C:/tp/video/vid{k}.mov')
    label = 0
    j = 0
    file = open(f'C:\\tp\\label{k}.txt', 'w+')
    img_array = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break


        if ret:
            img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            r = model.detect([img], verbose=0)
            r = r[0]
            
            new_img = mrcnn.visualize.display_instances(image=img,
                                    boxes=r['rois'],
                                    masks=r['masks'],
                                    class_ids=r['class_ids'],
                                    class_names=CLASS_NAMES,
                                    scores=r['scores'])

            img_array.append(new_img)

            mask_len = len(r['masks'][0][0])
            mask_count = 0
            index = None
            for i in range(mask_len):
                mask = r['masks'][:,:,i]
                temp = np.count_nonzero(mask == True)
                if temp > mask_count:
                    mask_count = temp
                    index = i

            try:
                a = np.where(r['masks'][960,:,index]==True)[0]
                label = (a[-1]+a[0])/2
                print('label :', label)
                file.write(f'{j}\t{label}\n')

            except:
                file.write(f'{j}\t{label}\n')

            j += 1

    file.close()

    print(np.array(img_array).shape)
    size = (720, 1280)

    out = cv2.VideoWriter(f'C:\\tp\\video\\video{k}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60.03, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    cap.release()
    out.release()
    cv2.destroyAllWindows()