import os
import codecs
import numpy as np
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC





X_sample = np.loadtxt('feature_array.txt')
#X_normal=normalize(X_sample, norm='max', axis=1, copy=True)

#Y_label=np.loadtxt('response_array.txt',fmt="%s")
Y_label=np.genfromtxt('response_array.txt',dtype=None)

X_train, X_test, y_train, y_test = train_test_split(X_sample, Y_label, test_size=0.33, random_state=42)	
#print Y_label

#differt svm settings and scores
"""clf=svm.SVC(decision_function_shape='ovo',C=22.68, gamma=0.88)
y_pred=clf.set_params(kernel='rbf').fit(X_train,y_train).predict(X_test)
print "ovo",accuracy_score(y_test, y_pred)


clf=svm.SVC(decision_function_shape='ovr',C=22.68, gamma=0.88)
y_pred=clf.set_params(kernel='rbf').fit(X_train,y_train).predict(X_test)
print "ovr",accuracy_score(y_test, y_pred)"""


#end differnt

#tune params

# Set the parameters by cross-validation on SVC
tuned_parameters = {'kernel': ['rbf'], 'gamma': [0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9],
                     'C': [1,5,10,15,20,25,30,35,40,45]}

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_weighted' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.
