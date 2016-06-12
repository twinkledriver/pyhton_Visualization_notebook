#**********************��һ�£�׼������**************************************
#*****************************************************************************************
#*****************************************************************************************
import requests
r=requests.get('https://github.com/timeline.json')
print r.content



#����matplotlib ���� ���� P11
import matplotlib.pyplot as plt
import numpy as np

t=np.arange(0.0,1.0,0.01)

s=np.sin(2*np.pi*t)
plt.rcParams['lines.color']='R'
plt.plot(t,s)

c=np.cos(2*np.pi*t)
plt.rcParams['lines.linewidth']='10'
plt.plot(t,c)

plt.show()

Ҳ������matplotlibrc �ļ��и��� Ĭ�����ã�������Ч��

#*****************************************************************************************
#******************************************�ڶ���*****************************************
#*****************************************************************************************


#******************************************��ȡCSV�ļ�*****************************************

��������


import csv
filename='3367OS_02_Code/ch02-data.csv'
data=[]

try:
    with open(filename) as f:
        reader = csv.reader(f)
        c = 0
        for row in reader:
            if c == 0:
                header = row
            else:
                data.append(row)
            c += 1
except csv.Error as e:
	print "Error reading CSV file at line %s : %s" % (reader.line_num,e)
	sys.exit(-1)

if header:
	print header
	print '================='

for datarow in data:
	print datarow


#ǿ��Ҫע�� ���� ��ʱ���۾� ����ʹ ��� ��չ ������ ˵���߼�



#******************************************��ȡMicrosoft Excel�ļ�*****************************************

#һ��һ�� һ��һ�� һ����Ԫһ����Ԫ�Ķ� ���

import xlrd

file = '3367OS_02_Code/ch02-xlsxdata.xlsx'

wb = xlrd.open_workbook(filename=file)

ws = wb.sheet_by_name('Sheet1')					#������sheet1

dataset = []

for r in xrange(ws.nrows):  # ��
	col = []
	for c in range(ws.ncols):  #��
		col.append(ws.cell(r,c).value)  #��Ԫ���ֵ
	dataset.append(col)					#׷��

from pprint import pprint
pprint(dataset)

#******************************************��ȡ���������ļ�*****************************************

#ͨ����ʽƥ�� ����ȡ����

ch02-generate_f_data.py �������� ��ʽ ��Sample �� ����

#������ ��ʽ�Ķ�ȡ����

import struct 
import string

datafile='3367OS_02_Code/ch02-fixed-width-1M.data'

mask='9s14s5s'

with open(datafile,'r') as f:
	for line in f:
		fields=struct.Struct(mask).unpack_from(line)
		print 'fields: ',[field.strip() for field in fields]


#******************************************��ȡ�Ʊ���ָ��ʽ�ļ�*****************************************

import csv
filename='3367OS_02_Code/ch02-data.tab'

data=[]

try:
    with open(filename) as f:
        reader = csv.reader(f,dialect=csv.excel_tab)  #   ������Ĭ��Ϊexcel��ʽ��Ҳ���Ƕ���(,)�ָ�������csvģ��Ҳ֧��excel-tab���Ҳ�����Ʊ��(tab)�ָ���
        c = 0
        for row in reader:
            if c == 0:
                header = row
            else:
                data.append(row)
            c += 1
except csv.Error as e:
	print "Error reading CSV file at line %s : %s" % (reader.line_num,e)
	sys.exit(-1)
if header:
	print header
	print '================='
for datarow in data:
	print datarow



#���� ������  ������� ���з�  ��Ҫ ����һ�� �˴���split()��ʵ ��Ϊ�� ���á�

import csv
filename='3367OS_02_Code/ch02-data-dirty.tab'

data=[]

try:
    with open(filename,'r') as f:
     for line in f:
		 line=line.strip()
		 print line.split()
except csv.Error as e:
	print "Error reading CSV file at line %s : %s" % (reader.line_num,e)
	sys.exit(-1)


#******************************************��ȡJSON��ʽ�ļ�*****************************************

import requests

url='https://github.com/timeline.json'

r=requests.get(url)
json_obj=r.json()

repos=set()

for entry in json_obj:
	try:
		repos.add(entry['repository']['url'])  #��һ�� ������  �����ǰ汾��һ����
	except KeyError as e:
		print "No key %s. Skipping. . ." %(e)

from pprint import pprint
pprint(repos)

#******************************************JSON������*****************************************
import json
jstring='{"name":"prod1","price":12.50}'
from decimal import Decimal
json.loads(jstring,parse_float=Decimal)

#******************************************�������ݸ�ʽ�ļ�**************************************
import xlwt

#1.���� ��Ҫ��ģ��
import os
import sys
import argparse
import struct
import json
import csv

try:
	import cStringIO as StringIO
except:
	import StringIO


#2.Ȼ�󣬶�����ʵĶ�д���ݵķ���  ��װ����

# ���ļ� ��װ
def import_data(import_file):
	mask='9s14s5s'
	data=[]
	with open(import_file,'r') as f:
		for line in f:
			fields=struct.Struct(mask).unpack_from(line)
			data.append(list([f.strip() for f in fields]))
	return data


#д�ļ� ��װ
def write_data(data,export_format):
	if export_format == 'csv':
		return write_csv(data)
	elif export_format == 'json':
		return write_json(data)
	elif export_format == 'xlsx':
		return write_xlsx(data)
	else:
		raise Exception("Illeagal format defined")

#3����ͬ���͵ĺ��� �ֱ�д

def write_csv(data):
	f = StringIO.StringIO()
	writer = csv.writer(f)
	for row in data:
		writer.writerow(row)
	return f.getvalue()

def write_json(data):
	j = jspn.dumps(data)
	return j

def write_xlsx(data):
	from xlwt import Workbook
	book = Workbook()
	sheet1 = book.add_sheet("Sheet 1")
	row = 0
	for line in data:
		col = 0
		for datum in line:
			print datum
			sheet1.write(row,col,datum)
			col += 1
		row += 1
		if row>65535:
			print>>sys.stderr,"Hit limit of # of rows in one sheet (65535)."
			break
		f = StringIO.StringIO()
		book.save(f)
		return f.getvalue()
	


# ����·�� �������� ������ʽ
if __name__  == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument("import_file", help="Path to a fixed-width data file.")
    parser.add_argument("export_format", help="Export format: json, csv, xlsx.")
    args = parser.parse_args()

if args.import_file is None:
	print >> sys.stderr,"You must specify path to impot from."
	sys.exit(1)

if args.export_file not in ('csv','json','xlsx'):
	print >> sys.stderr,"You must provide valid export file format."
	sys.exit(1)

if not os.path.isfile(args.import_file):
	print>>sys.stderr,"Given path is not a file :%s "%args.import_file
	sys.exit(1)

data=import_data(args.import_file)

print write_data(data,args.export_format)

			
#******************************************�����ݿ⵼������*****************************************
#��鰲װ�걸
import sqlite3
sqlite3.version
sqlite3.sqlite_version

#����
import sqlite3
import sys
                           
if len(sys.argv) < 2:
    print "Error: You must supply at least SQL script."
    print "Usage: %s table.db ./sql-dump.sql" % (sys.argv[0])       #��cmd ���� py�ļ� �������
    sys.exit(1)

script_path = sys.argv[1]

if len(sys.argv) == 3:
    db = sys.argv[2]
else:
    # if DB is not defined 
    # create memory database
    db = ":memory:"											# δָ��db�ļ��� ���� �ڴ��д���

try:
    con = sqlite3.connect(db)
    with con:
        cur = con.cursor()
        with open(script_path,'rb') as f:
            cur.executescript(f.read())
except sqlite3.Error as err:
    print "Error occured: %s" % err





import sqlite3								#��ѯ����
import sys

if len(sys.argv) != 2:
    print "Please specify database file."
    sys.exit(1)

db =  '3367OS_02_Code/test.db'

try:
    con = sqlite3.connect(db)
    with con:
        cur = con.cursor()
        query = 'SELECT ID, Name, Population FROM City ORDER BY ID  LIMIT 20'    #sql���

        con.text_factory = str
        cur.execute(query)

        resultset = cur.fetchall()							#��ȡ���н����

        # extract column names

        col_names = [cn[0] for cn in cur.description]    # ID    Name   Population   
        print "%10s %30s %10s" % tuple(col_names)   #ǰ�����ո�
        print "="*(10+1+30+1+10)

        for row in resultset:							
            print "%10s %30s %10s" % row										#���д�ӡ�����
except sqlite3.Error as err:													#�쳣����
    print "[ERROR]:", err

#******************************************�����쳣ֵ*****************************************

#��ͳ��ѧ �� MAD ��λ������ƫ�� �ķ���


import numpy as np
import matplotlib.pyplot as plt

def is_outlier(points,threshold=3.5):

	if len(points.shape) == 1:                                  #ʸ����
		points= points[:,None]

	median = np.median(points,axis = 0)             #��λ��

	diff=np.sum((points-median)**2,axis=-1)
	diff=np.sqrt(diff)																						#�����׼��

	med_abs_deviation = np.median(diff)

	modified_z_score = 0.6745 * diff / med_abs_deviation      #z-score���� ����㷨

	return modified_z_score > threshold                       #����boolֵ������������ݣ�

x = np.random.random(100)

buckets = 50

x=np.r_[x,-49,95,100,-100,200,90,300,500]                      #�쳣ֵ

filtered = x[~is_outlier(x)]                                              #�����ɾ�����

plt.figure()                                             #��ͼ

plt.subplot(211)
plt.hist(x,buckets)
plt.xlabel('Raw')

plt.subplot(212)
plt.hist(filtered,buckets)
plt.xlabel('Cleaned')

plt.show()

#���۹۲취 ����ͼ

from pylab import *

spread = rand(50) * 100
center  = ones(25) * 50

flier_high = rand(10) * 100 +100
flier_low = rand(10) * -100

data = concatenate((spread,center,flier_high,flier_low),0)

subplot(311)
boxplot(data,0,'gx')                             # ����ͼ  ��� ���� Ƶ��  x���� �쳣ֵ

subplot(312)

spread_1 = concatenate((spread,flier_high,flier_low),0)         #ɢ��ͼ  ����x ����25
center_1 = ones(70) *25
scatter(center_1,spread_1)
xlim([0,50])

subplot(313)
center_2 =rand(70) *50
scatter(center_2,spread_1)
xlim([0,50])

x = 1e6* rand(1000)
y = rand(1000)

figure()

subplot(211)

scatter(x,y)

xlim(1e-6,1e6)

subplot(212)

scatter(x,y)

xscale('log')

xlim(1e-6,1e6)



#******************************************��ȡ��������ļ�****************************************

# �� cmd ��  �� python ch02-chunk-read.py 'Ҫ�����ļ�'  �Ϳ��� �����ģ�� 

import sys

filename = sys.argv[1]

with open(filename,'rb') as hugefile:
	chunksize= 1000
	readable = ''
	start = hugefile.tell()        #tell��������ָ��
	print "starting at :",start

	file_block= ''
	for _ in xrange(start,start+chunksize):
		line = hugefile.next()
		file_block =file_block +line
		print 'file_block',type(file_block) ,file_block
	readable = readable +file_block

	stop = hugefile.tell()
	print 'readable',type(readable), readable
	print 'reading bytes from %s to %s '% (start,stop)
	print 'read bytes total:' , len(readable)

	raw_input()


#******************************************��ȡ�������ļ�****************************************	
import time
import os 
import sys

if len(sys.argv) != 2:
	print >> sys.stderr,"Please specify filename to read"
filename = sys.argv[1]

if not os.path.isfile(filename):
	print >> sys.stderr,"Given file: \"%s\" is not a file" % filename

with open(filename,'r') as f:
	filesize = os.stat(filename)[6]     #stat()������ filename��ָ���ļ�״̬�����Ƶ�����buf��ָ�Ľṹ��
	f.seek(filesize)									#��λ�� �ļ�β

	while True:												#ʼ��ѭ��
		where = f.tell()									#����ָ��λ��
		line = f.readline()
		if not line:
			time.sleep(1)
			f.seek(where)
		else:
			print line

		#cmd�� python ch02-stream-read-1.py �����仯���ļ�����


#******************************************����ͼ�����ݵ�numpy����****************************************	

#��lena ͼ��Ԥ���� misc �У�

import scipy.misc
import matplotlib.pyplot as plt

lena = scipy.misc.lena()

#plt.gray()

plt.imshow(lena)
plt.colorbar()
plt.show()

print lena,shape
print lena.max()
print lena.dtype

#���� stinkbug ͼ��

import numpy
import Image
import matplotlib.pyplot as plt

bug = Image.open('3367OS_02_Code/stinkbug.png')
arr =  numpy.array(bug.getdata(),numpy.uint8).reshape(bug.size[1],bug.size[0],3)

plt.gray()
plt.imshow(arr)
plt.colorbar()
plt.show()


#�Ŵ�ͼƬ

import matplotlib.pyplot as plt
import scipy
import numpy
import scipy.misc


bug = scipy.misc.imread('3367OS_02_Code/stinkbug.png')

print bug.shape

bug =bug[:,:,0]

plt.figure()
plt.gray()

plt.subplot(121)
plt.imshow(bug)

zbug = bug[100:350,140:350]

plt.subplot(122)
plt.imshow(zbug)
plt.show()




#******************************************���ɿɿص�������ݼ���****************************


#����һ���򵥵����������

import pylab
import random
SIZE=100

random.seed()

real_rand_vars=[]

real_rand_vars = [random.random() for var in xrange(SIZE)]
pylab.hist(real_rand_vars,10)

pylab.xlabel("Number range")
pylab.ylabel("Count")

pylab.show()

#��������۸�����

import pylab
import random

duration = 100
mean_inc =0.2

std_dev_inc = 1.2 
x= range(duration)
y=[]
price_today=0

for i in x:
	next_delta = random.normalvariate(mean_inc,std_dev_inc)
	price_today += next_delta
	y.append(price_today)

pylab.plot(x,y)
pylab.xlabel("Time")
pylab.ylabel("Value")
pylab.show()


#��ͬ�ķֲ�ģ��

import random
import matplotlib
import matplotlib.pyplot as plt

SAMPLE_SIZE= 1000
buckets =100

plt.figure()

matplotlib.rcParams.update({'font.size':7})



#0 1 ��������ֲ�
plt.subplot(621)
plt.xlabel("random.random")

res=[random.random() for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)


#���ȷֲ�

plt.subplot(622)
plt.xlabel("random.uniform")

a = 1
b = SAMPLE_SIZE

res=[random.uniform(a,b) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)

#���Ƿֲ�
plt.subplot(623)
plt.xlabel("random.triangular")


low = 1
high = SAMPLE_SIZE

res=[random.triangular(low,high) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)

#beta�ֲ�

plt.subplot(624)
plt.xlabel("random.beta")


alpha = 1
beta = 10

res = [random.betavariate(alpha,beta) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)

#ָ���ֲ�


plt.subplot(625)
plt.xlabel("random.expovariate")

lambd = 1.0/((SAMPLE_SIZE + 1) / 2.)
res=[random.expovariate(lambd) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)

#gamma �ֲ�


plt.subplot(626)
plt.xlabel("random.gamma")

alpha = 1
beta =10
res =[random.gammavariate(alpha,beta) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)

#������̫�ֲ�

plt.subplot(627)
plt.xlabel("random.lognormvariate")

mu =1
sigma =0.5

res= [random.lognormvariate(mu,sigma) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)


#��̫�ֲ�
plt.subplot(628)
plt.xlabel("random.normvariate")

mu =1
sigma =0.5

res= [random.normalvariate(mu,sigma) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)

#������ �ֲ�

plt.subplot(629)
plt.xlabel("ranodm.paretovariate")
alpha =1
res =[random.paretovariate(alpha) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)



#******************************************����ƽ������****************************


from pylab import *
from numpy import *

def moving_average(interval,window_size):
	window = ones(int(window_size))/float(window_size)
	return convolve(interval,window,'same')

t=linspace(-4,4,100)
y= sin(t) +randn(len(t)) *0.1

plot(t,y,"k.")

y_av=moving_average(y,10)
plot(t,y_av,"r")

xlabel("Time")
ylabel("Value")
grid(True)
show()


#******************************************���߼� ��ƽ������****************************

import numpy
from numpy import *
from numpy import *

WINDOWS=['flat','hanning','hamming','bartlett','blackman']

#WINDOW=['flat','hanning']

def smooth(x,window_len= 11 ,window ='hanning'):
	if x.ndim !=1:
		raise ValueError,"smooth only accepts 1 dimension arrays."
	if x.size < window_len:
		raise ValueError,"Input vector needs to be bigger than window size."
	if window_len<3:
		return x
	if not window in WINDOWS:
		raise ValueError("Window is one of '' 'flat','hanning','hamming','bartlett','blackman'")

	s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

	if window == 'flat': #moving average
		w = numpy.ones(window_len, 'd')
	else:
        # call appropriate function in numpy
		w = eval('numpy.' + window + '(window_len)')
		y = numpy.convolve(w/w.sum(), s, mode='valid')
		return y


t = linspace(-4, 4, 100)

# Make some noisy sinusoidal
x = sin(t)
xn = x + randn(len(t))*0.1

# Smooth it
y = smooth(x)

# windows
ws = 31

subplot(211)
plot(ones(ws))



# draw on the same axes
hold(True)

# plot for every windows
for w in WINDOWS[1:]:
    eval('plot('+w+'(ws) )')

# configure axis properties
axis([0, 30, 0, 1.1])

# add legend for every window
legend(WINDOWS)

title("Smoothing windows")

# add second plot
subplot(212)

# draw original signal 
plot(x)

# and signal with added noise
plot(xn)

# smooth signal with noise for every possible windowing algorithm
for w in WINDOWS:
    plot(smooth(xn, 10, w))

# add legend for every graph
l=['original signal', 'signal with noise']
l.extend(WINDOWS)
legend(l)

title("Smoothed signal")

show()


#*******************************��ֵ�˲�***************************************

import numpy as np 
import pylab as p
import scipy.signal as signal

x = np.linspace(0,1,101)

x[3::10] = 1.5
p.plot(x)
p.plot(signal.medfilt(x,3))
p.plot(signal.medfilt(x,5))
p.plot(signal.medfilt(x,7))
p.show()

#********************************************************************************
#***************************�����»��Ʋ����ƻ�ͼ��*********************************************
#********************************************************************************

plot([1,2,3,2,3,2,2,1])

plot([4,3,2,1],[1,2,3,4])

from matplotlib.pyplot import *
x = [1,2,3,4]
y = [5,4,3,2]

figure()

subplot(231)
plot(x,y)

subplot(232)
bar(x,y)

subplot(233)
barh(x,y)

subplot(234)
bar(x,y)
y1 = [7,8,5,3]
bar(x,y1,bottom =y,color ='r')

subplot(235)
boxplot(x)

subplot(236)
scatter(x,y)

#*************************�򵥵�����ͼ������ͼ***********************************

import matplotlib.pyplot as pl
import numpy as np

x = np.linspace(-np.pi,np.pi,256,endpoint=True)

y = np.cos(x)
y1= np.sin(x)

pl.plot(x,y)
pl.plot(x,y1)

pl.show()

xlim(-3.0,3.0)

#*************************����������ĳ��� �� ��Χ***********************************

axis()

l = [-1,1,-10,10]

axis(l)

axhline()
axvline(2)
axhline(4)

matplotlib.pyplot.grid()


#�̶�

from pylab import *

ax = gca()
ax.locator_params(tight = True,nbins =10)
ax.plot(np.random.normal(10,.1,100))

show()

