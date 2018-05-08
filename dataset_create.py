import lungc
import os.path
import cv2
import random
import numpy as np

nocanc=[]
nocanc_output =[]
for i in range(1,139):#without folder #run for full data
	if not os.path.isfile('without/nocanc ('+str(i)+').jpg'):
		continue
	print(i)
	img=cv2.imread('without/nocanc ('+str(i)+').jpg',0)
	left_feature,right_feature,img_left,img_right=lungc.process_lung(img)
	
	nocanc.append(left_feature)
	nocanc_output.append(0)

cancl=[]
cancl_output =[]
for i in range(1,121):#left folder #run for full data
	if not os.path.isfile('Left/canc ('+str(i)+').jpg'):
		continue
	print(i)
	img=cv2.imread('Left/canc ('+str(i)+').jpg',0)	
	left_feature,right_feature,img_left,img_right=lungc.process_lung(img)
	
	cancl.append(left_feature)
	cancl_output.append(1)

random.seed(2)
dataset=cancl+nocanc
dataset_output = cancl_output + nocanc_output 
random.shuffle(dataset)
random.shuffle(dataset_output)

with open('datafile.py', 'w') as f:
	f.write('dataset = %s' % dataset)
	f.write('\ndataset = %s' % dataset_output)
