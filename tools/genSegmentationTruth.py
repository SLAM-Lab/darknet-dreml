import os
import sys

path2darknet = '/home/osboxes/Documents/darknet_x86_dreml/'

if len(sys.argv) != 4:
	print('usage:',sys.argv[0],'[imagelist] [path_to_input_folder] [path_to_output_folder]')
	exit(1)

cfg_fn = 'instance_segment.cfg'
weights_fn = 'instance_segment_161000.weights'

inputDir = os.path.abspath(sys.argv[2])
outputDir = os.path.abspath(sys.argv[3])

os.system('rm -rf ' + outputDir)
os.system('mkdir ' + outputDir)

with open(sys.argv[1]) as fh:
	for line in fh:
		imName = line.split('.txt')[0]
		print(imName)
		os.system('cd ' + path2darknet + '/; ./darknet segmenter test cfg/maskyolo.data cfg/' + cfg_fn + ' ../darknet_weights/' + weights_fn + ' ' + inputDir + '/' + imName + '.jpg')
		os.system('cd ' + path2darknet + '/; mv iseg_result.txt ' + outputDir + '/' + imName + '.txt')
