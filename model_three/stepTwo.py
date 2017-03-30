#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from datetime import date
import os
import sys
import copy

# 24187 record the set of P |[sku_id,attr1,attr2,attr3,cate,brand]->>sk_id
product=pd.read_csv('../../data/JData_Product.csv')
print "P counts: len(product)"
P=product.iloc[:,0]

#combine data of neg,pos, get rate
filePath="../mydatas/traindata_interver_three_days/"
off_pos_neg=['pos','neg']
def rateData():
	for cc in off_pos_neg:
		fileName="off_train_"+cc+"2016-04-"
		name=filePath+fileName+"01.csv"
		print "--------FileName------ ",name
		negData=pd.read_csv(name)
		negData=negData[negData.col_sum!=0]
		print "----\]n", negData.head(5)
		for item in range(1,7):
			strType='type_%d_rate'%item
			sum_type='sum_type_%d'%item
			negData[strType]=negData[sum_type]/negData['col_sum']
		negDatas =negData[['user_id','sku_id','type_1_rate','type_2_rate','type_3_rate','type_4_rate','type_5_rate','type_6_rate','label']]

		for idx in range(2,9):
			name=filePath+fileName+"%02d.csv"%idx
			print "--------FileName------ ",name
			negData=pd.read_csv(name)
			negData=negData[negData.col_sum!=0]
			print "----\]n", negData.head(5)
			for item in range(1,7):
				strType='type_%d_rate'%item
				sum_type='sum_type_%d'%item
				negData[strType]=negData[sum_type]/negData['col_sum']
			negDataTemp =negData[['user_id','sku_id','type_1_rate','type_2_rate','type_3_rate','type_4_rate','type_5_rate','type_6_rate','label']]
			negDatas=pd.concat([negDatas,negDataTemp])


		print negDatas.head(10)
		print len(negDatas)
		negDatas.to_csv(filePath+"off_train_"+cc+"_rate_201604.csv",index=False)

def getFinalTestData0415():

    offTestData=pd.read_csv('../../data/JData_Action_201604.csv')
    print len(offTestData)
    indexs=[8]
    for myindex in indexs:
        startTime='2016-04-%02d'%myindex
        endTime='2016-04-%02d'%(myindex+2)
        offTestData=offTestData[(offTestData.time>=startTime)&(offTestData.time<endTime)&(offTestData['sku_id'].isin(P))].ix[:,[0,1,2,4]]
        print "----testData ---",len(offTestData),"record \n",offTestData.head(5)
        testFeaturesType_6=offTestData.ix[:,[0,1,3]]
        testFeaturesType_6['sum_type_6']=1
        testFeaturesType_6 = testFeaturesType_6[testFeaturesType_6['type']==6].iloc[:,[0,1,3]].groupby(['user_id','sku_id']).agg('sum').reset_index()
        testFeatures=copy.deepcopy(testFeaturesType_6)
        for indx in range(1,6):
          testFeaturesName='features_type_'+str(indx)
          print "-------features_name-----",testFeaturesName
          testFeaturesName=offTestData.ix[:,[0,1,3]]
          testFeaturesName['sum_type_'+str(indx)]=1
          testFeaturesName=testFeaturesName[testFeaturesName['type']==indx].iloc[:,[0,1,3]].groupby(['user_id','sku_id']).agg('sum').reset_index()
          testFeatures=pd.merge(testFeatures,testFeaturesName, on=['user_id','sku_id'],how='left')
        #t3.rename(columns={'date_received':'dates'},inplace=False)
        testFeatures=testFeatures.fillna(int(0))
        testFeatures=testFeatures.iloc[:,[0,1,3,4,5,7,2,6]]  
        testFeatures['col_sum']=testFeatures.ix[:,[2,3,4,5,6,7]].apply(lambda x: x.sum(),axis=1)
        offTestData=testFeatures
        '''
        #features.loc['row_sum']=features.iloc[:,2:9].apply(lambda x : x.sum())
        testFeatures=testFeatures[testFeatures['sum_type_4']>=1]
        testFeatures.to_csv("../mydatas/testFeatures"+str(myindex)+"-"+str(myindex+4)+".csv",index=False)
        print len(testFeatures)
        '''

    negData=offTestData[offTestData.col_sum!=0]
    print "----\]n", negData.head(5)
    for item in range(1,7):
        strType='type_%d_rate'%item
        sum_type='sum_type_%d'%item
        negData[strType]=negData[sum_type]/negData['col_sum']
    negDatas =negData[['user_id','sku_id','type_1_rate','type_2_rate','type_3_rate','type_4_rate','type_5_rate','type_6_rate']]
    '''# the case >1 day 
    for idx in range(2,3):
        name=filePath+fileName+"%02d.csv"%idx
        print "--------FileName------ ",name
        negData=pd.read_csv(name)
        negData=negData[negData.col_sum!=0]
        print "----\]n", negData.head(5)
        for item in range(1,7):
            strType='type_%d_rate'%item
            sum_type='sum_type_%d'%item
            negData[strType]=negData[sum_type]/negData['col_sum']
        negDataTemp =negData[['user_id','sku_id','type_1_rate','type_2_rate','type_3_rate','type_4_rate','type_5_rate','type_6_rate','label']]
        negDatas=pd.concat([negDatas,negDataTemp])
    '''
    print negDatas.head(10)
    print len(negDatas)
    negDatas.to_csv("../mydatas/testData/on_test_rate_8-10.csv",index=False)

if __name__ == '__main__':
	#rateData()
	getFinalTestData0415()


