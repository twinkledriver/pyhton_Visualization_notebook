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
