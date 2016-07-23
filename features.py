import cv2
import numpy as np

def find_feature(char):
	return np.array(zonewise_hu5(char)+htow_ratio(char)+find_blobs(char),np.float32)

def feature_hu2(img):
	contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	moments = [0,0,0,0,0,0,0,0,0,0,0,0]
	if(len(contours)==0):
		return moments
	X = [cv2.contourArea(C) for C in contours]
	t=[i for i in range (0,len(contours))]
	X,t = zip(*sorted(zip(X,t),reverse=True))
	list = []
	for i in range (0,2):
		try:
 #			print (i)
			cnt = contours[i]
			if(cv2.contourArea(cnt)<4):
				[list.append(0.0) for j in range(0,6)]
				continue
			mom = cv2.HuMoments(cv2.moments(cnt))
			moments=mom[:-1]
			[list.append(m[0]) for m in moments]
		except IndexError:
			[list.append(0.0) for j in range(0,6)]
	return list

def zonewise_hu5(img):#diagonal with more contours
	global ter
	contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	X = [cv2.contourArea(C) for C in contours]
	t=[i for i in range (0,len(contours))]
	try:
		X,t = zip(*sorted(zip(X,t),reverse=True))
	except ValueError:
		cv2.imwrite('error_temp.png',img)
		print 'no countours'
		exit
	cnt = contours[t[0]]
	x,y,w,h=cv2.boundingRect(cnt)
	for i in range(x,x+w):
		for j in range(y,y+h):
			if(cv2.pointPolygonTest(cnt,(i,j),False)==-1):
				img[j,i]=0
	im = img[y-1:y+h+1,x-1:x+w+1]
	height,width=im.shape
	box = img[0:1,0:1]
	box[0,0]=0
	box = cv2.resize(box,(width,height))
	img4=[]
	[img4.append(box.copy())for i in range(0,4)]
	i=0
	for i in range (0,height):
		j=(int)(i*width/height)
		for k in range(0,width):
			if(k<j):
				img4[0][i,k]=im[i,k]
				img4[0][height-i-1,k]=im[height-i-1,k]
			elif(k>width-j):
				img4[2][i,k]=im[i,k]
				img4[2][height-i-1,k]=im[height-i-1,k]
			else:
				img4[1][i,k]=im[i,k]
				img4[3][height-i-1,k]=im[height-i-1,k]
		if (j>width/2):
			break
	i=0
	feature = []
	for img in img4:
		feature = feature+feature_hu2(img)
	return feature
def find_blobs(im):
	params=cv2.SimpleBlobDetector_Params()
	params.filterByArea=True
	params.minArea=10
	params.filterByConvexity=True
	params.minConvexity=0.87
	detector=cv2.SimpleBlobDetector(params)
	keypoints=detector.detect(im)
	# print len(keypoints)
	return [len(keypoints)]
def htow_ratio(im):
	h,w=im.shape
	q=0
	for i in range(h):
		for j in range(w):
			if im.item(i,j)==255:
				q+=1
	# print [h/w,(q/(h*w))]
	return [h/w,(q/(h*w))]
