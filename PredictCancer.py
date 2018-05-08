import cv2
import lungc
import pickle

num=input("Enter test image number: ")
img=cv2.imread("test/test ("+num+").jpg",0)
left_feature,right_feature,img_left,img_right=lungc.process_lung_test(img)


filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict([left_feature])
print(result[0])
if(result[0]):
	print("Tumour present")
else:
	print("Normal Lung")
