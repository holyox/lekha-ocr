import os
import codecs
import cv2
import features
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix





X_sample = np.loadtxt('feature_array.txt')
#Y_label=np.loadtxt('response_array.txt',fmt="%s")
Y_label=np.genfromtxt('response_array.txt',dtype=None)

X_train, X_test, y_train, y_test = train_test_split(X_sample, Y_label, test_size=0.08, random_state=42)	
#print Y_label
#X_normal=normalize(X_sample, norm='max', axis=1, copy=True)


clf=svm.SVC(decision_function_shape='ovo',C=22.68, gamma=0.88)
y_pred=clf.set_params(kernel='rbf').fit(X_train,y_train).predict(X_test)

cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
#cma = np.array(cm,np.float32)
#np.savetxt('confusion.txt', cma)

print('Confusion matrix, without normalization')
print(cm)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
y_test1=[x.decode('UTF8') for x in y_test]
#print y_test1

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test1)))
    plt.xticks(tick_marks,(np.unique(y_test1)), rotation=0)
    plt.yticks(tick_marks,(np.unique(y_test1)))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure()
plot_confusion_matrix(cm)
plt.show()
