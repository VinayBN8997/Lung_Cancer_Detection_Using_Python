import cv2
from scipy import ndimage
import numpy as np
import os.path
from math import sqrt 

def preprocess(img):
	img_filted=cv2.medianBlur(img,5)#5 is mask size
	thresh,img_thresh=cv2.threshold(img_filted,127,255,cv2.THRESH_BINARY)
	kernel=np.ones((7,7),np.uint8)
	img_opened=cv2.morphologyEx(img_thresh,cv2.MORPH_OPEN,kernel)
	img_closed=cv2.morphologyEx(img_opened,cv2.MORPH_CLOSE,kernel)
	return img_closed,img_filted,img_thresh,img_opened

def lung_side_detect(c_index,contours,img):
	img_roied_left_lung=img
	img_roied_right_lung=img
	#left lung check
	x,y,w,h=cv2.boundingRect(contours[c_index[0]])
	M=cv2.moments(contours[c_index[0]])
	cx=int(M['m10']/M['m00'])
	cy=int(M['m01']/M['m00'])
	#print(cx,cy,int(img.shape[1]/2))
	if(cx<int(img.shape[1]/2)):
		img_roied_left_lung=img[y:y+h,x:x+w]
	else:
		img_roied_right_lung=img[y:y+h,x:x+w]
	#right lung check
	x,y,w,h=cv2.boundingRect(contours[c_index[1]])
	M=cv2.moments(contours[c_index[1]])
	cx=int(M['m10']/M['m00'])
	cy=int(M['m01']/M['m00'])
	#print(cx,cy,int(img.shape[1]/2))
	if(cx<int(img.shape[1]/2)):
		img_roied_left_lung=img[y:y+h,x:x+w]
	else:
		img_roied_right_lung=img[y:y+h,x:x+w]

	return img_roied_left_lung,img_roied_right_lung

def seg_cont(img):
	img_cont,contours,hierarchy=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	zipped=zip([i for i in range(len(contours))],[cv2.contourArea(contours[i]) for i in range(len(contours))])
	zipped=list(reversed(sorted(zipped,key=lambda x:x[1])))
	c_index=[zipped[1][0],zipped[2][0]]
	img_roied_left_lung,img_roied_right_lung=lung_side_detect(c_index,contours,img)
	return img_roied_left_lung,img_roied_right_lung

def sub_feature_extract(sub_image):
	feature_list = []
	height,width = sub_image.shape
	total_pix = height * width		
	sub_image_1 = sub_image[0:int(height/2),0:width]		
	sub_image_2 = sub_image[int(height/2):height,0:width]
	sub_image_3 = sub_image[0:height,0:int(width/2)] 
	sub_image_4 = sub_image[0:height,int(width/2):width]
	unique,counts=np.unique(sub_image_1,return_counts=True)#gives number of white and black pix
	feature_list.append(counts[0]/total_pix)#feature1=black_pix_up/tot_pix
	unique,counts=np.unique(sub_image_2,return_counts=True)#gives number of white and black pix
	feature_list.append(counts[0]/total_pix)#feature2=black_pix_down/tot_pix
	unique,counts=np.unique(sub_image_3,return_counts=True)#gives number of white and black pix
	feature_list.append(counts[0]/total_pix)#feature3=black_pix_left/tot_pix
	unique,counts=np.unique(sub_image_4,return_counts=True)#gives number of white and black pix
	feature_list.append(counts[0]/total_pix)#feature4=black_pix_right/tot_pix	
	return	feature_list

def sub_feature_extract_2(sub_image):
	feature_list = []
	height,width=sub_image.shape
	total_pix = height*width
	sub_image_1 = sub_image[0:int(height/2),0:int(width/2)]
	sub_image_2 = sub_image[0:int(height/2),int(width/2):width]
	sub_image_3 = sub_image[int(height/2):height,0:int(width/2)]
	sub_image_4 = sub_image[int(height/2):height,int(width/2):width]
	unique,counts=np.unique(sub_image_1,return_counts=True)#gives number of white and black pix
	feature_list.append(counts[0]/total_pix)#feature1=black_pix_up_left/tot_pix
	unique,counts=np.unique(sub_image_2,return_counts=True)#gives number of white and black pix
	feature_list.append(counts[0]/total_pix)#feature2=black_pix_up_right/tot_pix
	unique,counts=np.unique(sub_image_3,return_counts=True)#gives number of white and black pix
	feature_list.append(counts[0]/total_pix)#feature3=black_pix_down_left/tot_pix
	unique,counts=np.unique(sub_image_4,return_counts=True)#gives number of white and black pix
	feature_list.append(counts[0]/total_pix)#feature4=black_pix_down_right/tot_pix	
	return	feature_list

	
def feature_extract(lung_image):
	feature_vector=[]
	height,width=lung_image.shape#height is shape(0)
	#print(shape)
	cent_x=int(width/2)
	cent_y=int(height/2)
	feature_vector.append(height/width)#feature1=height/width
	total_pix=height*width
	feature_list = sub_feature_extract(lung_image)#feature2,3,4,5 
	feature_vector = feature_vector + feature_list
	feature_list = sub_feature_extract_2(lung_image)#feature 6,7,8,9
	feature_vector = feature_vector + feature_list
	feature_list = sub_feature_extract_2(lung_image[0:int(height/2),0:int(width/2)])#feature 10,11,12,13
	feature_vector = feature_vector + feature_list
	feature_list = sub_feature_extract_2(lung_image[0:int(height/2),int(width/2):width])#feature 14,15,16,17
	feature_vector = feature_vector + feature_list
	feature_list = sub_feature_extract_2(lung_image[int(height/2):height,0:int(width/2)])#feature 18,19,20,21
	feature_vector = feature_vector + feature_list
	feature_list = sub_feature_extract_2(lung_image[int(height/2):height,int(width/2):width])#feature 22,23,24,25
	feature_vector = feature_vector + feature_list
	
	feature_26_list = []	
	m00 = []
	m10 = []
	m01 = []
	for i in range(height): #y vale
		for j in range(width): #x value
			if lung_image[i,j] == 0:
				feature_26_list.append(sqrt(((j - cent_x)**2) + ((i - cent_y)**2)))
			m00.append(lung_image[i,j])
			m10.append(j * lung_image[i,j])
			m01.append(i * lung_image[i,j])
	feature_vector.append(sum(feature_26_list)/total_pix) #feature 26
	m00 = sum(m00)
	m10 = sum(m10)
	m01 = sum(m01)
	x_bar = m10/m00
	y_bar = m01/m00
	
	u = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
	for i in range(height): #y vale
		for j in range(width): #x value
			for p in range(4):
				for q in range(4):
					u[p][q].append(((j-x_bar)**p)*((i-y_bar)**q)*lung_image[i,j])	 
	neta = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
	for p in range(4):
		for q in range(4):
			neta[p][q] = sum(u[p][q])/((sum(u[0][0]))**(((p+q)/2)+1))
	feature_vector.append(neta[2][0] + neta[0][2]) #feature 27
	feature_vector.append((neta[2][0] - neta[0][2])**2 + 4*((neta[1][1])**2)) #feature 28
	feature_vector.append((neta[3][0] - 3*neta[1][2])**2 + (3*neta[2][1] - neta[0][3])**2) #feature 29
	feature_vector.append((neta[3][0] + neta[1][2])**2 + (neta[2][1] + neta[0][3])**2) #feature 30
	feature_vector.append(((neta[3][0]-3*neta[1][2])*(neta[3][0] + neta[1][2])*((neta[3][0] + neta[1][2])**2 - 3*((neta[2][1] + neta[0][3])**2))) + ((3*neta[2][1]-neta[0][3])*(neta[2][1] + neta[0][3])*(3*(neta[3][0] + neta[1][2])**2 - ((neta[2][1] + neta[0][3])**2)))) #feature 31
	feature_vector.append((neta[2][0] - neta[0][2])*(((neta[3][0] + neta[1][2])**2) - ((neta[2][1] + neta[0][3])**2)) + (4*neta[1][1]*(neta[3][0] + neta[1][2])*(neta[2][1] + neta[0][3]))) #feature 32
	feature_vector.append(((3*neta[2][1]-neta[0][3])*(neta[3][0] + neta[1][2])*((neta[3][0] + neta[1][2])**2 - 3*((neta[2][1] + neta[0][3])**2))) - ((neta[3][0] + 3*neta[1][2])*(neta[2][1] + neta[0][3])*(3*(neta[3][0] + neta[1][2])**2 - ((neta[2][1] + neta[0][3])**2)))) #feature 33
	
	return feature_vector
	 
def process_lung(img):
	r=30
	img=img[r:512-r,r:512-r]
	img_closed,img_filted,img_thresh,img_opened=preprocess(img)
	img_roied_left_lung,img_roied_right_lung=seg_cont(img_closed)
	feature_vector_left = feature_extract(img_roied_left_lung)
	feature_vector_right = 1#feature_extract(img_roied_right_lung)
	return feature_vector_left,feature_vector_right,img_roied_left_lung,img_roied_right_lung#img,img_closed,img_filted,img_thresh,img_opened,img_roied_left_lung,img_roied_right_lung,feature_vector
	
def process_lung_test(img):
	r=30
	img=img[r:512-r,r:512-r]
	img_closed,img_filted,img_thresh,img_opened=preprocess(img)
	img_roied_left_lung,img_roied_right_lung=seg_cont(img_closed)
	feature_vector_left = feature_extract(img_roied_left_lung)
	feature_vector_right = 1#feature_extract(img_roied_right_lung)

	cv2.imshow('img_filted',img_filted)
	cv2.waitKey()
	cv2.imshow('img_thresh',img_thresh)
	cv2.waitKey()
	cv2.imshow('img_opened',img_opened)
	cv2.waitKey()
	cv2.imshow('img_closed',img_closed)
	cv2.waitKey()
	cv2.imshow('img_roied_left_lung',img_roied_left_lung)
	cv2.waitKey()
	cv2.imshow('img_roied_right_lung',img_roied_right_lung)
	cv2.waitKey()
	cv2.destroyAllWindows()
	
	return feature_vector_left,feature_vector_right,img_roied_left_lung,img_roied_right_lung
	#img,img_closed,img_filted,img_thresh,img_opened,img_roied_left_lung,img_roied_right_lung,feature_vector
ace care about it\

for i in [k]: #replace k by the list of images u need to train with... Image name format could differ.Take care about it.
	if not os.path.isfile('Left/canc ('+str(i)+').jpg'):
		continue
	print(i)
	img=cv2.imread('Left/canc ('+str(i)+').jpg',0)	
	####
	r=30
	img=img[r:512-r,r:512-r]

	img_closed,img_filted,img_thresh,img_opened=preprocess(img)
	img_roied_left_lung,img_roied_right_lung=seg_cont(img_closed)
	####
	cv2.imshow('img_filted',img_filted)
	cv2.waitKey()
	cv2.imshow('img_thresh',img_thresh)
	cv2.waitKey()
	cv2.imshow('img_opened',img_opened)
	cv2.waitKey()
	cv2.imshow('img_closed',img_closed)
	cv2.waitKey()
	####
	cv2.imshow('img_roied_leftlung1',img_roied_left_lung)
	cv2.waitKey()
	cv2.imshow('img_roied_rightlung2',img_roied_right_lung)
	cv2.waitKey()
	cv2.destroyAllWindows()
	feature_vector = feature_extract(img_roied_left_lung)
	print(len(feature_vector))
	####
	features=process_lung(img)
	print(len(features))


