import imageio as im
from matplotlib import pyplot as pt
import numpy as np
#read image and plot histogram
image = im.imread("crowd.tif")
y=image.shape
row = y[0]
col=y[1]
count = [0]*256
for i in range(0,row):
    for j in range(0,col):
        count[image[i,j]]=count[image[i,j]]+1
        
ab = list(range(0,256))
pt.bar(ab,count)

#check if 2 peaks are present
cd = list(range(0,16))
grouped = [0]*16
for i in range(0,16):
    grouped[i] = count[16*i]+count[16*i+15]

c=0
for i in range(1,15):
    if grouped[i-1]+300<grouped[i] and grouped[i]>grouped[i+1]+300 :
        c=c+1

if grouped[0]>grouped[1]:
    c=c+1
if grouped[15]>grouped[14]:
    c=c+1    

if c==2:
    print("Foreground: Yes")
else:
    print("Foreground: No")