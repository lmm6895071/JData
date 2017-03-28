# -*- encoding:utf8 -*-

import os
import  time
import MySQLdb
import json
import pandas as pd
import numpy as np
from datetime import date
import copy

# 24187 record the set of P |[sku_id,attr1,attr2,attr3,cate,brand]->>sk_id
product=pd.read_csv('../../data/JData_Product.csv')
print "P counts: len(product)"
P=product.iloc[:,0]
GLOBAL_P=list(P.get_values())

def test():
    offTestData=pd.read_csv('../../data/JData_Action_201604.csv')
    print len(offTestData)
    offTestData=offTestData[(offTestData.time>='2016-04-10')&(offTestData.time<'2016-04-11')&(offTestData['sku_id'].isin(P))].ix[:,[0,1,2,4]]
    print "----testData ---",len(offTestData),"record \n"
    testFeaturesType_6=offTestData.ix[:,[0,1,3]]
    testFeaturesType_6['sum_type_6']=1
    testFeaturesType_6 = testFeaturesType_6[testFeaturesType_6['type']==6].iloc[:,[0,1,3]].groupby(['user_id','sku_id']).agg('sum').reset_index()
    testFeatures=copy.deepcopy(testFeaturesType_6)
    for index in range(1,6):
      testFeaturesName='features_type_'+str(index)
      print "-------features_name-----",testFeaturesName
      testFeaturesName=offTestData.ix[:,[0,1,3]]
      testFeaturesName['sum_type_'+str(index)]=1
      testFeaturesName=testFeaturesName[testFeaturesName['type']==index].iloc[:,[0,1,3]].groupby(['user_id','sku_id']).agg('sum').reset_index()
      testFeatures=pd.merge(testFeatures,testFeaturesName, on=['user_id','sku_id'],how='left')
    testFeatures=testFeatures.fillna(int(0))
    testFeatures=testFeatures.iloc[:,[0,1,3,4,5,7,2,6]]  
    testFeatures['col_sum']=testFeatures.ix[:,[2,3,4,5,6,7]].apply(lambda x: x.sum(),axis=1)
    testFeatures=testFeatures[(testFeatures.sum_type_4>=0)&(testFeatures.sum_type_2>=1)&(testFeatures.sum_type_2-testFeatures.sum_type_3>0)]
    testFeatures= testFeatures[testFeatures.sku_id.isin(P)].groupby('user_id').apply(lambda t: t[t.col_sum==t.col_sum.max()])

    testFeatures=testFeatures.groupby('user_id').apply(lambda x: x)
    predata=testFeatures.ix[:,[0,1]]
    getF(predata)
   
#get the set of P(products)
def getF(predata):
	testdata=pd.read_csv("../mydatas/windows/testFeatures11-15.csv")
	realSums=len(testdata)
	print "realSums=",len(testdata)

	A1=float(len(predata[predata['user_id'].isin(testdata.ix[:,0])]))
	tempData=pd.merge(predata,testdata,on=['user_id','sku_id'],how='left')
	tempData=tempData.dropna()
	A2=float(len(tempData))
	sumsUids=len(predata.ix[:,0])
	realSumsUids=len(list(set((testdata.ix[:,0]).get_values())))
	print realSumsUids
	P1=A1/sumsUids
	R1=A1/realSumsUids # user's counts (buy goods)
	P2=A2/sumsUids
	R2=A2/realSums
	print "A1=",A1,"\tA2=",A2
	print "R1=",R1,"\tR2=",R2
	print "P1=",P1,"\tP2=",P2
	f11=0
	f12=0
	try:
		f11=6*R1*P1/(5*R1+P1)
		f12=5*R2*P2/(2*R2+3*P2)
	except:
		pass

	score=0.4*f11+0.6*f12
	print "f11=",f11,"\tf12=",f12,"\nfinal----score=",score,

if __name__ == '__main__':
	#for x in range(10,16):
	#getConnetion(x)
	dic = {'a':31, 'bc':5, 'c':3, 'asd':4, 'aa':74, 'd':0}
	dictw= sorted(dic.iteritems(), key=lambda d:d[1], reverse = True)
	print dictw
	test()