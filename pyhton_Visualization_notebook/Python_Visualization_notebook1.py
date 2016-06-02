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

for entry in js