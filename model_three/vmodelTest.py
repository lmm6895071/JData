# -*- coding:utf-8 -*-

import sys
import os
reload(sys)
sys.setdefaultencoding('utf8')

import re
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from numpy import *
from sklearn import metrics
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale,normalize
import time
import pandas as pd 
import numpy as np

import score

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))

class VModel:
    posCount=0
    negCount=0
    def __init__(self):
        pass
    def getPLData(self):
        #path = "../mydatas/traindata_interver_one_days/"
        path = "../mydatas/traindata_interver_three_days/"
        posD=pd.read_csv(path+"off_train_pos_rate_201604.csv",header=0)
        negD=pd.read_csv(path+"off_train_neg_rate_201604.csv",header=0)
        rdata_pos=[]
        rdata_neg=[]

        rdata_pos=[[row[col_name] for col_name in posD.columns[2:7] ] for index, row in posD.iterrows()]
        rdata = [[row[col_name] for col_name in negD.columns[2:7] ]for index ,row in negD.iterrows() ]
        
        posCount=len(rdata_pos)*5
        negCount=posCount
        shuffleArray = range(len(rdata))
        np.random.shuffle(shuffleArray)
        for ii in xrange(negCount):
            rdata_neg.append(rdata[shuffleArray[ii]])

        y = np.concatenate((np.ones(posCount), np.zeros(negCount)))
        X = np.concatenate((rdata_pos, rdata_pos))
        for ii in range(1,4):
            X = np.concatenate((X, rdata_pos))
        X = np.concatenate((X, rdata_neg))
        print X[0:10],"\n",y[0:10]
        print len(X),len(y)
        return  (X,y)

    def myPredict(self):
       
        #clf = joblib.load("../mydatas/outModel/model1/Random Forest.m")
        ml_list=["Decision Tree.m","Nearest Neighbors.m","Random Forest.m","LogisticRegression.m","GradientBoostingClassifier.m"]
        for item in ml_list:
            preData=pd.read_csv("../mydatas/testData/on_test_rate_10.csv",header=0)
            rdata=[[row[col_name] for col_name in preData.columns[2:7] ] for index, row in preData.iterrows()]
            clf = joblib.load("../mydatas/outModel/model1/"+item)
            pre_y=clf.predict(rdata)
            preData['label']=pre_y
            preData= preData[preData['label']==1].ix[:,0:2]
            print "\n#######"+item+"######\nthis end data lines:",len(preData)
            #preData.to_csv('result/model1_result.csv',index=False)
            combineResult(preData)
    def testModel(self,X,y,size=0.2):
        classifiers = [
            KNeighborsClassifier(),
            #SVC(kernel="linear", C=0.025),
            #SVC(gamma=2, C=1),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            LogisticRegression(),
            GradientBoostingClassifier()
        ]
        names = ["Nearest Neighbors", 
             "Decision Tree", "Random Forest", "AdaBoost",
             "LogisticRegression", "GradientBoostingClassifier"]#"Linear SVM", "RBF SVM",

        #classifiers = [RandomForestClassifier()]
        #names=[" Random Forest"]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size)
        for name,clf in zip(names,classifiers):
            print "---------",name,"------------"
            startT = time.time()
            clf.fit(X_train,y_train)
            endT = time.time()
            y_true,y_pred = y_test,clf.predict(X_test)
            startT = time.time()
            #print (clf.coef_)
            print (classification_report(y_true,y_pred))
            print (metrics.confusion_matrix(y_true,y_pred))
            joblib.dump(clf,"../mydatas/outModel/model2/"+name+".m")

def combineResult(pre_result):
    rule_result=pd.read_csv("result/rule_result_10.csv")
    print "\nrule result:------",len(rule_result)
    score.getF(rule_result)
    print "\nmodel1 result:------",len(pre_result)
    score.getF(pre_result)
    
    result=pd.concat([rule_result,pre_result])
    result=result.drop_duplicates(['user_id'])
    print "\ncobine result",len(result)
    score.getF(result)
    #result.to_csv("result/result.csv",index=False)
    


if __name__ == "__main__":
    model =VModel()
    '''
    x,y=model.getPLData()
    model.testModel(x,y)
    '''
    pre_result=model.myPredict()
    




