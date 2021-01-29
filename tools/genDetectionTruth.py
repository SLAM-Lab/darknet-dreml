import os
import sys

path2darknet = '/home/osboxes/Documents/darknet_x86_dreml/'

if len(sys.argv) != 5:
        print('usage:',sys.argv[0],'[neuralnet] [imagelist] [path_to_input_folder] [path_to_output_folder]')
        exit(1)

net = sys.argv[1]
currDir = os.getcwd()

assert net in ['yolov2','yolov3','yolov2-tiny','yolov3-tiny']

inputDir = os.path.abspath(sys.argv[3])
outputDir = os.path.abspath(sys.argv[4])

os.system('rm -rf '+outputDir)
os.system('mkdir '+outputDir)

with open(sys.argv[2],'r') as fh:
	for line in fh:
		imName = line.split('.txt')[0]
		print(imName)
		os.system('cd ' + path2darknet + '/; ./darknet detect ./cfg/' + net + '.cfg ../darknet_weights/' + net + '.weights ' + inputDir + '/' + imName + '.jpg')
		os.system('cd ' + path2darknet + '/; mv mAP_result.txt ' + outputDir + '/' + imName + '.txt')
