# -*- encoding:utf8 -*-

import os
import  time
import MySQLdb
import json
import pandas as pd
import numpy as np
from datetime import date
import copy
import score
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
    testFeatures=testFeatures[(testFeatures.sum_type_4<1)&(testFeatures.sum_type_2>=1)&(testFeatures.sum_type_2-testFeatures.sum_type_3>0)]
    testFeatures= testFeatures[testFeatures.sku_id.isin(P)].groupby('user_id').apply(lambda t: t[t.col_sum==t.col_sum.max()])

    testFeatures=testFeatures.groupby('user_id').apply(lambda x: x)
    predata=testFeatures.ix[:,[0,1]]
    predata.to_csv("result/rule_result_10.csv",index=False)
    return predata
   

if __name__ == '__main__':
	#for x in range(10,16):
	#getConnetion(x)
	dic = {'a':31, 'bc':5, 'c':3, 'asd':4, 'aa':74, 'd':0}
	dictw= sorted(dic.iteritems(), key=lambda d:d[1], reverse = True)
	print dictw
	predata=test()
	score.getF(predata)
