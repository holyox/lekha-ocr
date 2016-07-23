Original Work is https://gitlab.com/space-kerala/lekha-OCR.git

THE DISCRIPTION GIVEN BELOW MAY CHANGE AS THE DEVELOPMENT PROGRESSES                                                                  
This Project is under Development and Code is buit in that way ,so is subject to errors


    
*Note: All modules are written in python. For image processing opencv is used, for training SVM of Scikit lern used
# lekha-OCR

Printed text recognizer for Malayalam, Lekha OCR is an optical character recognizer trained for the recognition of printed malayalam Documents.

Prerequirements
======
>OpenCV 2.4.11 (not works if opencv 3.0+)
>
>Python 2.7.9
>
>numpy (>= 1.6.1)
>
>scikit-learn

In debian 8 jessie and ubuntu 14.04
-----------
       #apt-get insatll python-opencv python-pip
       #pip install numpy scikit-learn
Usage
=======
To run this Project

    1>Exectute the training.py,The files used for training is in original repository(https://gitlab.com/space-kerala/lekha-OCR-database.git)

    2>Run the lekha.py .
      NOTE it doesnt take command line arguments.you have to change the file read path in lekha.py

    3>the output will be saved as test.txt

You can view the CONFUSION MATRIX buy running confusion.py for a given settings and parameters of svm

    1>the training set is split into test and train

Experiments can be done using Experiment.py where grid search and parameter optimisation techniques are tested.
    You can view the accuracy score for your tested parameter setings

Supporters
=======
This project is developed by [SPACE] (http://www.space-kerala.org/) in association with [ICFOSS] (http://icfoss.in/).

Contributors
=======
**Arun Joseph** contributed most of the engine initial developments.

**Sachin** developing second phase

**Jithin Thankachan** contributed some additional features, training tool and helped in documentation.

**Balagopal Unnikrishnan** contributed in preparing XML label for training and helped in documentation.

**Rijoy V** contributed in initial research.

**Ambily Sreekumar** contributed in building data set for training.

**Arun M** helped in project management and technical assistance.

This project is initially hosted at **[Github] (https://github.com/space-kerala/lekha_OCR_1.0)**
From 2016/April onwards new developments updating to gitlab.

***********************************************************************

***********************************************************************

