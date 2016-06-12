#**********************第一章：准备工作**************************************
#*****************************************************************************************
#*****************************************************************************************
import requests
r=requests.get('https://github.com/timeline.json')
print r.content



#配置matplotlib 参数 举例 P11
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

也可以在matplotlibrc 文件中更改 默认配置（长久有效）

#*****************************************************************************************
#******************************************第二章*****************************************
#*****************************************************************************************


#******************************************读取CSV文件*****************************************

导入数据


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


#强烈要注意 缩进 有时候眼睛 不好使 左边 扩展 有助于 说明逻辑



#******************************************读取Microsoft Excel文件*****************************************

#一行一行 一列一列 一个单元一个单元的读 表格

import xlrd

file = '3367OS_02_Code/ch02-xlsxdata.xlsx'

wb = xlrd.open_workbook(filename=file)

ws = wb.sheet_by_name('Sheet1')					#读的是sheet1

dataset = []

for r in xrange(ws.nrows):  # 行
	col = []
	for c in range(ws.ncols):  #列
		col.append(ws.cell(r,c).value)  #单元格的值
	dataset.append(col)					#追加

from pprint import pprint
pprint(dataset)

#******************************************读取定宽数据文件*****************************************

#通过格式匹配 来提取数据

ch02-generate_f_data.py 可以生成 格式 像Sample 的 代码

#下面是 正式的读取操作

import struct 
import string

datafile='3367OS_02_Code/ch02-fixed-width-1M.data'

mask='9s14s5s'

with open(datafile,'r') as f:
	for line in f:
		fields=struct.Struct(mask).unpack_from(line)
		print 'fields: ',[field.strip() for field in fields]


#******************************************读取制表符分割格式文件*****************************************

import csv
filename='3367OS_02_Code/ch02-data.tab'

data=[]

try:
    with open(filename) as f:
        reader = csv.reader(f,dialect=csv.excel_tab)  #   编码风格，默认为excel方式，也就是逗号(,)分隔，另外csv模块也支持excel-tab风格，也就是制表符(tab)分隔。
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



#对于 脏数据  多出几个 换行符  就要 处理一下 此处用split()其实 更为简单 适用。

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


#******************************************读取JSON格式文件*****************************************

import requests

url='https://github.com/timeline.json'

r=requests.get(url)
json_obj=r.json()

repos=set()

for entry in json_obj:
	try:
		repos.add(entry['repository']['url'])  #这一句 有问题  可能是版本不一样了
	except KeyError as e:
		print "No key %s. Skipping. . ." %(e)

from pprint import pprint
pprint(repos)

#******************************************JSON化处理*****************************************
import json
jstring='{"name":"prod1","price":12.50}'
from decimal import Decimal
json.loads(jstring,parse_float=Decimal)

#******************************************导出数据格式文件**************************************
import xlwt

#1.导入 需要的模块
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


#2.然后，定义合适的读写数据的方法  封装起来

# 读文件 封装
def import_data(import_file):
	mask='9s14s5s'
	data=[]
	with open(import_file,'r') as f:
		for line in f:
			fields=struct.Struct(mask).unpack_from(line)
			data.append(list([f.strip() for f in fields]))
	return data


#写文件 封装
def write_data(data,export_format):
	if export_format == 'csv':
		return write_csv(data)
	elif export_format == 'json':
		return write_json(data)
	elif export_format == 'xlsx':
		return write_xlsx(data)
	else:
		raise Exception("Illeagal format defined")

#3个不同类型的函数 分别写

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
	


# 解析路径 导入数据 导出格式
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

			
#******************************************从数据库导入数据*****************************************
#检查安装完备
import sqlite3
sqlite3.version
sqlite3.sqlite_version

#举例
import sqlite3
import sys
                           
if len(sys.argv) < 2:
    print "Error: You must supply at least SQL script."
    print "Usage: %s table.db ./sql-dump.sql" % (sys.argv[0])       #用cmd 调用 py文件 传入参数
    sys.exit(1)

script_path = sys.argv[1]

if len(sys.argv) == 3:
    db = sys.argv[2]
else:
    # if DB is not defined 
    # create memory database
    db = ":memory:"											# 未指定db文件名 就在 内存中创建

try:
    con = sqlite3.connect(db)
    with con:
        cur = con.cursor()
        with open(script_path,'rb') as f:
            cur.executescript(f.read())
except sqlite3.Error as err:
    print "Error occured: %s" % err





import sqlite3								#查询尝试
import sys

if len(sys.argv) != 2:
    print "Please specify database file."
    sys.exit(1)

db =  '3367OS_02_Code/test.db'

try:
    con = sqlite3.connect(db)
    with con:
        cur = con.cursor()
        query = 'SELECT ID, Name, Population FROM City ORDER BY ID  LIMIT 20'    #sql语句

        con.text_factory = str
        cur.execute(query)

        resultset = cur.fetchall()							#获取所有结果集

        # extract column names

        col_names = [cn[0] for cn in cur.description]    # ID    Name   Population   
        print "%10s %30s %10s" % tuple(col_names)   #前面留空格
        print "="*(10+1+30+1+10)

        for row in resultset:							
            print "%10s %30s %10s" % row										#按行打印结果集
except sqlite3.Error as err:													#异常处理
    print "[ERROR]:", err

#******************************************清理异常值*****************************************

#用统计学 中 MAD 中位数绝对偏差 的方法


import numpy as np
import matplotlib.pyplot as plt

def is_outlier(points,threshold=3.5):

	if len(points.shape) == 1:                                  #矢量化
		points= points[:,None]

	median = np.median(points,axis = 0)             #中位数

	diff=np.sum((points-median)**2,axis=-1)
	diff=np.sqrt(diff)																						#计算标准差

	med_abs_deviation = np.median(diff)

	modified_z_score = 0.6745 * diff / med_abs_deviation      #z-score计算 借鉴算法

	return modified_z_score > threshold                       #返回bool值（有问题的数据）

x = np.random.random(100)

buckets = 50

x=np.r_[x,-49,95,100,-100,200,90,300,500]                      #异常值

filtered = x[~is_outlier(x)]                                              #保留干净数据

plt.figure()                                             #绘图

plt.subplot(211)
plt.hist(x,buckets)
plt.xlabel('Raw')

plt.subplot(212)
plt.hist(filtered,buckets)
plt.xlabel('Cleaned')

plt.show()

#人眼观察法 箱线图

from pylab import *

spread = rand(50) * 100
center  = ones(25) * 50

flier_high = rand(10) * 100 +100
flier_low = rand(10) * -100

data = concatenate((spread,center,flier_high,flier_low),0)

subplot(311)
boxplot(data,0,'gx')                             # 箱线图  框宽 代表 频率  x代表 异常值

subplot(312)

spread_1 = concatenate((spread,flier_high,flier_low),0)         #散点图  所有x 都是25
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



#******************************************读取大块数据文件****************************************

# 在 cmd 中  用 python ch02-chunk-read.py '要读的文件'  就可以 用这个模块 

import sys

filename = sys.argv[1]

with open(filename,'rb') as hugefile:
	chunksize= 1000
	readable = ''
	start = hugefile.tell()        #tell（）返回指针
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


#******************************************读取数据流文件****************************************	
import time
import os 
import sys

if len(sys.argv) != 2:
	print >> sys.stderr,"Please specify filename to read"
filename = sys.argv[1]

if not os.path.isfile(filename):
	print >> sys.stderr,"Given file: \"%s\" is not a file" % filename

with open(filename,'r') as f:
	filesize = os.stat(filename)[6]     #stat()用来讲 filename所指的文件状态，复制到参数buf所指的结构中
	f.seek(filesize)									#定位到 文件尾

	while True:												#始终循环
		where = f.tell()									#返回指针位置
		line = f.readline()
		if not line:
			time.sleep(1)
			f.seek(where)
		else:
			print line

		#cmd中 python ch02-stream-read-1.py ‘“变化的文件名”


#******************************************导入图像数据到numpy数组****************************************	

#打开lena 图（预存在 misc 中）

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

#处理 stinkbug 图像

import numpy
import Image
import matplotlib.pyplot as plt

bug = Image.open('3367OS_02_Code/stinkbug.png')
arr =  numpy.array(bug.getdata(),numpy.uint8).reshape(bug.size[1],bug.size[0],3)

plt.gray()
plt.imshow(arr)
plt.colorbar()
plt.show()


#放大图片

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




#******************************************生成可控的随机数据集合****************************


#生成一个简单的随机数样本

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

#生成虚拟价格增长

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


#不同的分布模型

import random
import matplotlib
import matplotlib.pyplot as plt

SAMPLE_SIZE= 1000
buckets =100

plt.figure()

matplotlib.rcParams.update({'font.size':7})



#0 1 随机变量分布
plt.subplot(621)
plt.xlabel("random.random")

res=[random.random() for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)


#均匀分布

plt.subplot(622)
plt.xlabel("random.uniform")

a = 1
b = SAMPLE_SIZE

res=[random.uniform(a,b) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)

#三角分布
plt.subplot(623)
plt.xlabel("random.triangular")


low = 1
high = SAMPLE_SIZE

res=[random.triangular(low,high) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)

#beta分布

plt.subplot(624)
plt.xlabel("random.beta")


alpha = 1
beta = 10

res = [random.betavariate(alpha,beta) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)

#指数分布


plt.subplot(625)
plt.xlabel("random.expovariate")

lambd = 1.0/((SAMPLE_SIZE + 1) / 2.)
res=[random.expovariate(lambd) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)

#gamma 分布


plt.subplot(626)
plt.xlabel("random.gamma")

alpha = 1
beta =10
res =[random.gammavariate(alpha,beta) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)

#对数正太分布

plt.subplot(627)
plt.xlabel("random.lognormvariate")

mu =1
sigma =0.5

res= [random.lognormvariate(mu,sigma) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)


#正太分布
plt.subplot(628)
plt.xlabel("random.normvariate")

mu =1
sigma =0.5

res= [random.normalvariate(mu,sigma) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)

#帕累托 分布

plt.subplot(629)
plt.xlabel("ranodm.paretovariate")
alpha =1
res =[random.paretovariate(alpha) for _ in xrange(1,SAMPLE_SIZE)]
plt.hist(res,buckets)



#******************************************数据平滑处理****************************


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


#******************************************更高级 的平滑处理****************************

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


#*******************************中值滤波***************************************

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
#***************************第三章绘制并定制化图表*********************************************
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

#*************************简单的正弦图和余弦图***********************************

import matplotlib.pyplot as pl
import numpy as np

x = np.linspace(-np.pi,np.pi,256,endpoint=True)

y = np.cos(x)
y1= np.sin(x)

pl.plot(x,y)
pl.plot(x,y1)

pl.show()

xlim(-3.0,3.0)

#*************************设置坐标轴的长度 和 范围***********************************

axis()

l = [-1,1,-10,10]

axis(l)

axhline()
axvline(2)
axhline(4)

matplotlib.pyplot.grid()


#刻度

from pylab import *

ax = gca()
ax.locator_params(tight = True,nbins =10)
ax.plot(np.random.normal(10,.1,100))

show()

