#!/usr/bin/python

import	os
import	sys

if len(sys.argv) != 2:
	print 'usage:',sys.argv[0],'[net]'
	exit(1)

net = sys.argv[1]
path2img = './data/dog.jpg'

assert net in ['yolov2','yolov3','yolov2-tiny','yolov3-tiny']

os.system('./darknet detect cfg/'+net+'.cfg ../darknet_weights/'+net+'.weights '+path2img)
