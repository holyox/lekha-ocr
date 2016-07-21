import cv2
import numpy as np
from collections import namedtuple
from sklearn import svm
from sklearn.externals import joblib
from math import atan2, degrees, pi
import features
import make_word

im = cv2.imread('/home/holyox/code/ml_imag/o11b.tiff',0)
img= cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,243,43)
classifier = joblib.load('/home/holyox/code/svm/svm/svm_data.lekha')
MyStruct = namedtuple('MyStruct', 'line_no line_start line_end bowl_start bowl_end word_no char_pos x1 y1 wi hi label')
NodeDb = []
height,width=img.shape
sump=np.zeros(height)
sumvp=np.zeros(width)
string=' '

#skew_start
"""edges = cv2.Canny(img,50,150,apertureSize = 3)
lines = cv2.HoughLinesP(edges,1,np.pi/180,200,minLineLength=25,maxLineGap=20)
avg=0
for k in range(0,len(lines[0])-1):
    x1,y1,x2,y2=lines[0][k]
    rads = atan2((y1-y2),(x2-x1))
    rads %= 2*pi
    degs = 360-degrees(rads)
    avg=avg+degs
avg =float(avg)/(k+1)
image_center = tuple(np.array(img.shape)/2)
rot_mat = cv2.getRotationMatrix2D(image_center,degs,1.0)
img = cv2.warpAffine(img, rot_mat,(width,height),flags=cv2.INTER_LINEAR)
cv2.imshow('ske',img)
cv2.waitKey(0)"""
#skew_end


hor_pix_den=[0 for i in range(0,height)]
for i in range(0,height):
	for j in range(0,width):
		sump[i]=sump[i]+img[i][j]
	sump[i]/=255
        if sump[i]<9:
            sump[i]=0
noise =8
fin=0
lineno=0
start=0
for i in range (2,height):
	if(sump[i]>noise and sump[i-1]<(noise) and (sump[i-2]<noise)):
            if(fin==0):
                fin=1
                start = i
        elif(fin==1 and (i+2)<height and (sump[i+1]<noise and sump[i+2]<noise) and (i-start)>13) :
            fin=0
	    end=i
            wordno=0
	    l1=0
	    l2=0
	    precount=0
	    for f in range(start,end):
    	      
    	      k=np.absolute(precount-sump[f])
    	      if ((sump[f]>precount) and k>l1):
        		l1=k
        		b1=f
    	      elif((precount>sump[f]) and k>l2):
        		l2=k
        		b2=f
	      precount=sump[f]
            Node = MyStruct(line_no=lineno,line_start=start,line_end=end,bowl_start=b1,bowl_end=b2,word_no=None,char_pos=None,x1 =None,y1 =None,wi =None,hi =None,label=None)
            lineno+=1
            
#            cv2.imshow('window',img[start:i,0:width])
#            cv2.waitKey(0)
            startw=0
            fin1=0
            word_space=(end-start)/5
            for p in range(0,width):
	        for q in range(start,end):
		    sumvp[p]=sumvp[p]+img[q][p]
                sumvp[p]=int(sumvp[p]/255)
                if sumvp[p]<=2:
                    sumvp[p]=0
                #print sumvp[p]
           
            for p in range(0,width-8):
                arr=np.array(sumvp[p-10:p])
                arr1=np.array(sumvp[p:p+10])
                #print arr
                if (fin1==0 and  np.count_nonzero(arr)<=1 and np.count_nonzero(arr1)>=7):
                    startw=p
                    #cv2.line(im,(startw,start),(startw,end),(0,0,0),2)
                    fin1=1
                elif(fin1==1 and p-startw>=40 and np.count_nonzero(arr)>=7 and np.count_nonzero(arr1)<=1):
                    Node=Node._replace(word_no=wordno)
                    wordno+=1
                    endw=p
                    #cv2.line(im,(startw,start),(startw,end),(0,0,0),2)
                    fin1=0
                    #cv2.line(im,(endw,start),(endw,end),(0,0,0),2)
		    t=img[start:end,startw:endw].copy()
                    contours2, hierarchy = cv2.findContours(t.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	            if(len(contours2)==0):
                        print 'no line'
	            contours = []
	            for cnt in contours2:
		        if(cv2.contourArea(cnt)>6):
			    try:
				contours.append(cnt)
			    except ValueError:
				print ('error')
				pass
	            Mset = [cv2.moments(cnt) for cnt in contours]
	            X = [int(M['m10']/M['m00']) for M in Mset]
	            index = [i for i in range(0,len(contours2))]
	            try:
		        X,index = zip(*sorted(zip(X,index)))
	            except:
		        h=1
                    char_list=[]
	            for i in index:
		        cnt = contours[i]
                        x,y,w,h=cv2.boundingRect(cnt)
		    	Node= Node._replace(char_pos=i)
		    	Node= Node._replace(x1=x)
		    	Node= Node._replace(y1=y)
		    	Node= Node._replace(wi=w)
		    	Node= Node._replace(hi=h)
			char=t[y-1:y+h+1,x-1:x+w+1]

			char_feature=np.array(features.find_feature(char.copy()),np.float32)
			letter=classifier.predict(char_feature.reshape(1,-1))
			#print letter[0]
			#.encode('utf-8')
                        Node= Node._replace(label=letter[0].decode('utf-8'))
                        NodeDb.append(Node)
                        char_list.append(letter[0])
                    #print char_list
                    string=string+make_word.form_word(char_list)
                    string=string+' '
            string=string+'\n'
f = open('test2', 'w')
f.write(string.encode('utf8'))
f.close()
#cv2.imwrite("out.tiff", im)
#for i in range(0,len(NodeDb)-1):
#    print NodeDb[i]

