
#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from datetime import date
import os
import sys
import copy

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

