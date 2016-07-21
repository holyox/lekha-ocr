import os
import codecs
import cv2
import features
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.preprocessing import normalize




X_sample = np.loadtxt('feature_array.txt')
#Y_label=np.loadtxt('response_array.txt',fmt="%s")
Y_label=np.genfromtxt('response_array.txt',dtype='str')
X_normal=normalize(X_sample, norm='max', axis=1, copy=True)


clf=svm.SVC(decision_function_shape='ovo',C=22.68, gamma=0.88)
clf.set_params(kernel='rbf').fit(X_normal,Y_label)
joblib.dump(clf, 'svm/svm_data.lekha')



clf=svm.SVC(decision_function_shape='ovo',C=22.68, gamma=0.88)
clf.set_params(kernel='rbf').fit(X_normal,Y_label)
joblib.dump(clf, 'svm/svm_data.lekha')

