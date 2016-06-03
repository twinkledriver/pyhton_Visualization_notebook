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


#使用
import_data('3367OS_02_Code/ch02-fixed-width-1M.data')

			

