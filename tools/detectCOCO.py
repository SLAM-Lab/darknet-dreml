#!/usr/bin/python

import os

imageDir = 'data/coco/images/val2014/'

with open('/home/osboxes/Documents/TinyYolo_omnetpp/scripts/valSubset2.txt') as fh:
	for line in fh:
		imName = line.split('.txt')[0]
		print imName
		os.system('./darknet detect cfg/yolov2.cfg ../darknet_weights/yolov2.weights ' + imageDir + imName + '.jpg')
		os.system('mv mAP_result.txt results/'+imName+'.txt')
