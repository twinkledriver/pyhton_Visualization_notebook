#**********************第四章：更多图表和定制化**************************************
#*****************************************************************************************
#*****************************************************************************************

#阴影效果

import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np

data = np.random.randn(70)

fontsize = 18 
plt.plot(data)

title ="This is figure title"
x_label ="This is x axis label"
y_label ="This is y axis label"

title_text_obj =plt.title(title,fontsize=fontsize,verticalalignment ='bottom')

title_text_obj.set_path_effects([patheffects.withSimplePatchShadow()])

offset_xy = (1,-1)
rgbRed =(1.0,1.0,1.0)
alpha =0.5

pe = patheffects.withSimplePatchShadow(offset_xy = offset_xy,shadow_rgbFace = rgbRed,patch_alpha =alpha)

xlabel_obj =plt.xlabel(x_label,fontsize =fontsize,alpha =0.5)
xlabel_obj.set_path_effects([pe])

ylabel_obj =plt.ylabel(y_label,fontsize=fontsize,alpha =0.5)
ylabel_obj.set_path_effects([pe])

plt.show()

#*********************为图表线条添加阴影**************************************
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

def setup(layout =None):
	assert layout is not None

	fig = plt.figure()
	ax = fig.add_subplot(layout)
	return fig,ax

def get_signal():
	t = np.arange(0.,2.5,0.01)
	s = np.sin(5* np.pi*t)
	return t,s

def plot_signal(t,s):
	line, =axes.plot(t,s,linewidth =5 ,color ='magenta')
	return line,

def make_shadow(fig,axes,line,t,s):
	delta = 2/72.
	offset =transform.ScaledTranslation(delta,-delta,fig.dpi_scale_trans)
	offset_transform = axes.transData + offset 
	axes.plot(t,s,linewifth=5,color ='gray',transform = offset_transform,zorder =0.5*line.get_zoder())


#没错。这个地方的定义，是这样的：
#
#一个.py文件，如果是自身在运行，那么他的__name__值就是"__main__"；
#
#如果它是被别的程序导入的（作为一个模块），比如：
#import re
#那么，他的__name__就不是"__main__"了。
#
#所以，在.py文件中使用这个条件语句，可以使这个条件语句块中的命令只在它独立运行时才执行

if __name__ == "__main__":  
	fig.axes =setup(111)
	t,s =get_signal()
	line, =plot_signal(t,s)

	make_shadow(fig,axes,line,t,s)

	axes.set_title('Shadow effet using an offert transform')
	plt.show()


#*********************为图表添加数据表**************************************
import numpy as np
import matplotlib.pyplot as plt

plt.figure()
ax = plt.gca()
y = np.random.randn(9)

col_labels = ['col1','col2','col3']
row_labels = ['row1','row2','row3']
table_vals =[[11,12,13],[21,22,23],[28,29,30]]
row_colors=['red','gold','green']

my_table = plt.table(cellText=table_vals,colWidths=[0.1]*3,rowLabels = row_labels,colLabels=col_labels,rowColours=row_colors,loc='upper right')
plt.plot(y)
plt.show()
#*********************使用子区**************************************

import matplotlib.pyplot as plt

plt.figure(0)
axes1 = plt.subplot2grid((3,3),(0,0),colspan = 3)
axes2 = plt.subplot2grid((3,3),(1,0),colspan = 2)
axes3 = plt.subplot2grid((3,3),(1,2))
axes4 = plt.subplot2grid((3,3),(2,0))
axes5 = plt.subplot2grid((3,3),(2,1),colspan = 2)

all_axes = plt.gcf().axes
for ax in all_axes:
	for ticklabel in ax.get_xticklabels() + ax.get_yticklabels():
		ticklabel.set_fontsize(10)

plt.suptitle("Demo of subplot2grid")
plt.show()

fig = plt.figure()
axes = fig.add_subplot(111)
rect = matplotlib.patches.Rectangle((1,1),width=6 ,height =12)
axes.add_patch(rect)
axes.figure.canvas.draw()

#*********************定制化网格**************************************
plt.plot([1,2,3,3.5,4,4.3,3])
plt.grid()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.cbook import get_sample_data

def get_demo_image():
    f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
    # z is a numpy array of 15x15
    Z = np.load(f)
    return Z, (-3, 4, -4, 3)

def get_grid(fig=None , layout=None, nrows_ncols=None):
	assert fig is not None
	assert layout is not None
	assert nrows_ncols is not None

	grid =ImageGrid(fig,layout,nrows_ncols=nrows_ncols,axes_pad=0.05,add_all=True,label_mode="L")

	return grid

def load_image_to_grid(grid,Z,*images):
	min,max =Z.min(),Z.max()
	for i,image in enumerate(image):
		axes =grid[i]
		axes.imshow(image,origin ="lower" , vmin=min,vmax =max,interpolation = "nearest")

if __name__ == "__main__":
	fig = plt.figure(1,(8,6))
	grid =get_grid(fig,111,(1,3))
	Z,extent = get_demo_image()

	image1 = Z
	image2 = Z[:,:10]
	image3 = Z[:,10:]

	load_images_to_grid(grid,Z,image1,image2,image3)

	plt.draw()
	plt.show()



#*********************绘制等高线**************************************

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def process_signals(x,y):
	return (1-(x**2+y**2)*np.exp(-y**3/3))

x = np.arange(-1.5,1.5,0.1)
y = np.arange(-1.5,1.5,0.1)

X,Y = np. meshgrid(x,y)

Z = process_signals(X,Y)

N = np.arange(-1,1.5,0.3)

CS = plt.contour(Z,N,linewidths = 2 , cmap = mpl.cm.jet)
plt.clabel(CS,inline=True,fmt='%1.1f',fontsize = 10)
plt.colorbar(CS)

plt.title('My function: $z=(1-x^2+y^2) e^{-(y^3)/3}$')
plt.show()


#*********************填充图表底层区域**************************************

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

t = range(1000)
y = [sqrt(i) for i in t]
plt.plot(t,y,color = 'red',lw =2)
plt.fill_between(t,y,color= 'silver')
plt.show()

#另外一个例子


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0,2,0.01)
y1 = np.sin(np.pi*x)
y2 = np.sin(np.pi*4*x)

fig = plt.figure()
axes1 = fig.add_subplot(211)
axes1.plot(x,y1,x,y2,color = 'grey')
axes1.fill_between(x,y1,y2,where = y2 <= y1,facecolor = 'blue' , interpolate =True)
axes1.fill_between(x,y1,y2,where = y2 >= y1,facecolor = 'gold' , interpolate =True)
axes1.set_title('Blue                        Gold-color')
axes1.set_ylim(-2,2)

y2 =np.ma.masked_greater(y2,0.8)
axes2 = fig.add_subplot(212,sharex =axes1)
axes2.plot(x,y1,x,y2,color ='black')
axes2.fill_between(x,y1,y2,where=y2<=y1,facecolor = 'blue' , interpolate =True)
axes2.fill_between(x,y1,y2,where=y2>=y1,facecolor = 'gold' , interpolate =True)
axes2.set_title('Masked')
axes2.set_ylim(-2,2)
axes2.grid('on')

plt.show()

#*********************绘制极线图**************************************
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

figsize =7
colormap =lambda r:cm.Set2(r/20.)
N = 18 

fig = plt.figure(figsize=(figsize,figsize))
ax = fig.add_axes([0.2,0.2,0.7,0.7],polar =True)

theta =np.arange(0.0,2*np.pi,2*np.pi/N)
radii = 20 * np.random.randn(N)
width = np.pi/4*np.random.rand(N)
bars =ax.bar(theta,radii,width=width,bottom =0.0)

for r,bar in zip(radii,bars):
	bar.set_facecolor(colormap(r))
	bar.set_alpha(0.6)
plt.show()



#**********************第五章：创建3D可视化图表**************************************
#*****************************************************************************************
#*****************************************************************************************

import random

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['font.size'] = 10
fig =plt.figure()
ax = fig.add_subplot(111,projection = '3d')

for z in[2011,2012,2013,2014]:
	xs =xrange(1,13)
	ys = 1000 * np.random.rand(12)

	color =plt.cm.Set2(random.choice(xrange(plt.cm.Set2.N)))
	ax.bar(xs,ys,zs=z,zdir='y',color=color,alpha=0.8)

ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xs))
ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ys))

ax.set_xlabel('Month')
ax.set_ylabel('Year')
ax.set_zlabel('Sales Net [usd]')

plt.show()

#*****************************绘制双曲面抛物线**********************************
import random

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

n_angles =36 
n_radii = 8

radii = np.linspace(0.125,1.0,n_radii)

angles = np.linspace(0,2*np.pi,n_angles,endpoint =False)
angles = np.repeat(angles[...,np.newaxis],n_radii,axis=1)

x = np.append(0,(radii*np.cos(angles)).flatten())
y = np.append(0,(radii*np.sin(angles)).flatten())

z = np.sin(-x * y)
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(x,y,z,cmap=cm.jet,linewidth=0.2)

plt.show()


#*****************************绘制3D直方图**********************************

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['font.size'] = 10

samples =25

x = np.random.normal(5,1,samples)
y = np.random.normal(3,.5,samples)

fig = plt.figure()
ax = fig.add_subplot(211,projection='3d')

hist,xedges,yedges = np.histogram2d(x,y,bins=10)

elements = (len(xedges)-1)*(len(yedges)-1)
xpos,ypos =np.meshgrid(xedges[:-1]+.25,yedges[:-1]+.25)

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(elements)

dx = .1*np.ones_like(zpos)
dy = dx.copy()

dz = hist.flatten()

ax.bar3d(xpos,ypos,zpos,dx,dy,dz,color = 'b',alpha =0.4)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

ax2 =fig.add_subplot(212)
ax2.scatter(x,y)
ax2.set_xlabel('X Axis')
ax2.set_ylabel('Y Axis')
plt.show()


#*****************************创建动画**********************************

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()
ax = plt.axes(xlim=(0,2),ylim=(-2,2))
line, = ax.plot([] ,[] ,lw =2)

def init():
	line.set_data([],[])
	return line,

def animate(i):
	x = np.linspace(0,2,1000)
	y = np.sin(2*np.pi*(x-0.01*i))*np.cos(22*np.pi*(x-0.01*i))
	line.set_data(x,y)
	return line,

animator = animation.FuncAnimation(fig,animate,init_func = init,frames =5,interval =20 ,blit= True)

animator.save('basic_animation.mp4',fps=30,extra_args=['-vcodec','libx264'],writer='ffmepg')
plt.show()

#*****************************使用OpenGL制作动画**********************************

import numpy
from mayavi.mlab import *

安装太麻烦了 略过。

#**********************第六章：用图像和地图绘制表格**************************************
#*****************************************************************************************
#*****************************************************************************************

#******************************************绘制带图像的图表********************************

import matplotlib.pyplot as plt
from matplotlib._png import read_png
from matplotlib.offsetbox import TextArea,OffsetImage,AnnotationBbox

def load_data():
	import csv
	with open('pirates_temperature.csv','r') as f:
		reader = csv.reader(f)
		header = reader.next()
		datarows =[]
		for row in reader:
			datarows.append(row)
		return header,datarows

def format_data(datarows):
	years,temps,pirates = [],[],[]
	for each in datarows:
		years.append(each[0])
		temps.append(each[1])
		pirates.append(each[2])
	return years,temps,pirates

if __name__ == "__main__":
	fig = plt.figure(figsize = (16,8))
	ax =plt .subplot(111)
	header,datarows =load_data()
	xlabel, ylabel, _ = header
	years,temperature,pirates =format_data(datarows)
	title ="Global Average Temperature vs. Number of Pirates"
	
	plt.plot(years,temperature,lw =2)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	for x in xrange(len(years)):
		xy =years[x],temperature[x]

		ax.plot(xy[0],xy[1],"ok")

		pirate =read_png('tall-ship.png')

		zoomc =int(pirates[x])*(1/90000.)

		imagebox =OffsetImage(pirate,zoom=zoomc)

		ab = AnnotationBbox(imagebox,xy,xybox=(-200.*zoomc,200.*zoomc),xycoords='data',boxcoords="offset points",pad = 0.1, arrowprops =dict(arrowstyle = "->",connectionstyle ="angle ,angleA=0,angleB=-30,rad=3"))
		ax.add_artist(ab)

		no_pirates =TextArea(pirates[x],minimumdescent=False)
		ab =AnnotationBbox(no_pirates,xy,xybox=(50.,-25.),xycoords='data',boxcoords="offset points",pad=0.3,arrowprops=dict(arrowstyle ="->",connectionstyle="angle,angleA=0,angleB=-30,rad=3"))
		ax.add_artist(ab)

	plt.grid(1)
	plt.xlim(1800,2020)
	plt.ylim(14,16)
	plt.show(title)

	plt.show()



#******************************************提取RGB图像************************************

import matplotlib.pyplot as plt
import matplotlib.image as mplimage
import matplotlib as mpl
import os

class ImageViewer(object):
    def __init__(self, imfile):
        self._load_image(imfile)
        self._configure()


        self.figure = plt.gcf()
        t = "Image: {0}".format(os.path.basename(imfile))
        self.figure.suptitle(t, fontsize=20)

        self.shape = (3, 2)


    def _configure(self):
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['figure.autolayout'] = False
        mpl.rcParams['figure.figsize'] = (9, 6)
        mpl.rcParams['figure.subplot.top'] = .9

    def _load_image(self, imfile):
        self.im = mplimage.imread(imfile)

    @staticmethod
    def _get_chno(ch):
        chmap = {'R': 0, 'G': 1, 'B': 2}
        return chmap.get(ch, -1)


    def show_channel(self, ch):
        bins = 256
        ec = 'none'
        chno = self._get_chno(ch)
        loc = (chno, 1)
        ax = plt.subplot2grid(self.shape, loc)
        ax.hist(self.im[:, :, chno].flatten(), bins, color=ch, ec=ec,label=ch, alpha=.7)
        ax.set_xlim(0, 255)
        plt.setp(ax.get_xticklabels(), visible=True)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklines(), visible=True)
        plt.setp(ax.get_yticklines(), visible=False)
        plt.legend()
        plt.grid(True, axis='y')
        return ax

    def show(self):
        loc = (0, 0)
        axim = plt.subplot2grid(self.shape, loc, rowspan=3)
        axim.imshow(self.im)
        plt.setp(axim.get_xticklabels(), visible=False)
        plt.setp(axim.get_yticklabels(), visible=False)
        plt.setp(axim.get_xticklines(), visible=False)
        plt.setp(axim.get_yticklines(), visible=False)

        axr = self.show_channel('R')
        axg = self.show_channel('G')
        axb = self.show_channel('B')

        plt.show()


if __name__ == '__main__':
    im = 'images/yellow_flowers.jpg'
    try: 
        iv = ImageViewer(im)
        iv.show()
    except Exception as ex:
        print ex
