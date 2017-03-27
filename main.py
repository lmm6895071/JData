# -*- encoding:utf8 -*-

import os
import  time
import MySQLdb
import json

GLOBAL_DATA={}
GLOBAL_P=[]
WEIGHT_DAYS=[1,0.95,0.9,0.8,0.75,0.6,0.5,0.5,0.5,0.4]
WEIGHT_TYPE={"1":0.09,"2":0.9,"3":-0.9,"4":-0.8,"5":0.5,"6":0.005}
def getConnetion(index):
	dt = "2016-04-%02d"%index
	print dt,"\t","action"+dt+".txt"
	fs =open("action"+dt+".txt","w")
	conn = MySQLdb.connect(host="localhost",port=3306,user="root",passwd="!QAZ2wsx",db="JData")
	cur = conn.cursor()
	sql="select substring(time,1,10),user_id,sku_id,type,count(*) from JDataAction4Table where substring(time,1,10)='%s' group by substring(time,1,10),user_id,sku_id,type order by user_id,sku_id,type "%dt
	cc=cur.execute(sql)
	print cc
	info = cur.fetchmany(cc)
	for ii in info:
		print (ii)
		fs.write(json.dumps(ii)+"\n")
	fs.close()
	cur.close()
	conn.commit()
	conn.close()


#get the rank table
def initData():
	index = 1
	for index in range(1,11):
		dt = "2016-04-%02d"%(index+5)
		fs = open("action"+dt+".txt","r")
		ls = fs.readlines()
		print dt,"\tlines= ",len(ls)
		fs.close()
		for item in ls:
			try:
				its = eval(item)
				k=its[1]+":"+its[2]
				if k in GLOBAL_DATA:
					v=GLOBAL_DATA[k]
				else:
					v=0
				tp=its[3]
				vs=v+WEIGHT_DAYS[10-index]*WEIGHT_TYPE[tp]*its[4]
				GLOBAL_DATA[k]=vs
			except Exception as err:
				print "error----",str(err)
				return
	fw = open("trains.txt","w")
	GLOBAL_DATAs=sorted(GLOBAL_DATA.items(), key=lambda d:d[1],reverse = True) 
	
	result_L=[]
	result_P=[]
	global GLOBAL_P
	lens=len(GLOBAL_DATA.keys())
	print "total counts of results without filters",lens
	paremates =100000
	print "paremates =",paremates
	
	fsult = open("result.csv","w")
	strs="user_id,sku_id\n"
	fsult.write(strs.encode("utf8"))
	
	for item in GLOBAL_DATAs[0:paremates]:
		#print item[0],item[1]
		k=item[0].split(":")
		if k[0] not in result_L and k[1] in GLOBAL_P:
			result_L.append(k[0])
			result_P.append(k[1])
			strs=k[0]+"\t"+k[1]+"\n"
			fsult.write(strs.encode("utf8"))
			fw.write(item[0])
			fw.write("\t"+repr(item[1])+"\n")
			
	fw.close()
	fsult.close()
	print "my results lines=",len(result_L)
	return (result_L,result_P)

# init test data for F,R,P
def initTest():
	testData=[]
	global GLOBAL_P
	conn = MySQLdb.connect(host="localhost",port=3306,user="root",passwd="!QAZ2wsx",db="JData")	
	cur = conn.cursor()
	sql="select distinct user_id, sku_id from JDataAction4Table where substring(time,1,10)>='2016-04-11' and type='4'"
	cc=cur.execute(sql)
	info = cur.fetchmany(cc)
	for ii in info:
		if ii[1] not in GLOBAL_P:
			continue
		value=ii[0]+":"+ii[1]
		testData.append(value)
	testData=list(set(testData))
	cur.close()
	conn.commit()
	conn.close()
	return testData
#get the set of P(products)
def initP():
	global GLOBAL_P
	fs=open("data/JData_Product.csv","r")
	ls=fs.readline()
	print ls
	ls=fs.readlines()
	fs.close()
	for item in ls:
		its = item.split(",")
		GLOBAL_P.append(its[0])
# get result of sunmit
def getF(resultL,resultP):
	global GLOBAL_P
	testData=initTest()
	print "testdata lines :",len(testData)
	A1=0.0;
	A2=0.0;
	sums=len(resultL)
	uids=[]
	pids=[]
	for it in testData:
		its=it.split(":")
		uids.append(its[0])
		pids.append(its[1])
	for index in range(1,sums):
		k=resultL[index]+":"+resultP[index]
		if k in testData:
			A2=A2+1
		if resultL[index] in uids:
			A1=A1+1
	P1=A1/sums
	R1=A1/4395.0 #4395 user's counts (buy goods)
	f11=6*R1*P1/(5*+P1)
	P2=A2/sums
	R2=A2/len(testData)
	f12=5*R2*P2/(2*R2+3*P2)
	score=0.4*f11+0.6*f12
	print "final----score=",score,"\nf11=",f11,"\tf12=",f12


if __name__ == '__main__':
	#for x in range(10,16):
	#getConnetion(x)
	dic = {'a':31, 'bc':5, 'c':3, 'asd':4, 'aa':74, 'd':0}
	dictw= sorted(dic.iteritems(), key=lambda d:d[1], reverse = True)
	print dictw
	initP()
	(resultL,resultP)=initData()
	getF(resultL,resultP)
