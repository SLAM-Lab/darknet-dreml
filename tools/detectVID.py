#!/usr/bin/python

import os
import sys

if len(sys.argv) != 2:
	print sys.argv[0], '[object]'
	exit(1)

obj = sys.argv[1]
imageDir = '/home/osboxes/Documents/vid_dataset/'+obj

with open(imageDir+'/'+obj+'_vid.txt') as fh:
	for line in fh:
		imName = line.split('.txt')[0]
		print imName
		os.system('./darknet detect cfg/yolov2.cfg ../darknet_weights/yolov2.weights ' + imageDir + '/images/' + imName + '.JPEG')
		os.system('mv mAP_result.txt results/'+imName+'.txt')
