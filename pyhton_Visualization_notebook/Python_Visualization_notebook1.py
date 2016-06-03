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


#ʹ��
import_data('3367OS_02_Code/ch02-fixed-width-1M.data')

			

