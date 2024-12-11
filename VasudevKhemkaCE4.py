import imageio.v2 as im
import numpy as np
from matplotlib import pyplot as plt

part = input('part a or b? \n')
if part == 'a':
    name = 'fingerprint.tif'
elif part == 'b':
    name = 'polymersomes.tif'

    #histogram generation
img=im.imread(name)
a=[0]*256
height,width=img.shape
for i in range(width):
    for j in range(height):
        g=img[j,i]
        a[g]=a[g]+1
ab= np.array(range(0,256))
plt.bar(ab,a)

# global thresholding

Tg = 0.33 * img.max()
Tg = Tg.astype(np.uint8)
del_Tg = 11
while del_Tg >5:
    binary_img = np.zeros((height,width))
    c1 = 0
    i = 0
    c2 = 0
    j = 0
    for (pixel,group) in np.nditer((img,binary_img), op_flags=['readwrite']):
        if pixel < Tg:
            group[...] = 0
            c1 = c1 + pixel
            i+=1
        elif pixel >=Tg:
            group[...] = 255
            c2 = c2 + pixel
            j+=1
    if i*j == 0:
        Tg_next = 1.5*Tg
    else:
        Tg_next = (c1/i + c2/j)/(2)
    Tg_next = Tg_next.astype(np.uint8)
    del_Tg = abs(Tg_next - Tg)
    Tg = Tg_next

print('global thresholding threshold:', Tg)
binary_img = binary_img.astype(np.uint8)
if part == 'a':
    im.imsave('VasudevKhemkaCE4aglobal.tif', binary_img)
elif part == 'b':
    im.imsave('VasudevKhemkaCE4bglobal.tif', binary_img)

#otsu
P = height*width
val_max = -1
for t in range(0,256):
    q1 = float(np.sum(a[:t]))/P
    q2 = float(np.sum(a[t:]))/P
    if q1 == 0:
        continue
    if q2 == 0:
        continue

    m1 = float(np.sum(np.array(list(range(t)))*a[:t])/q1)
    m2 = float(np.sum(np.array(list(range(t,256)))*a[t:])/q2)
    val = q1*(1-q1)*np.power(m1-m2,2)
    if val_max < val:
        val_max = val
        T_otsu = t

    binary_img_otsu = np.zeros((height,width))

    if pixel < Tg:
        group[...] = 0

    elif pixel >=Tg:
        group[...] = 255
binary_img_otsu = binary_img_otsu.astype(np.uint8)
if part == 'a':
    im.imsave('VasudevKhemkaCE4aotsu.tif', binary_img_otsu)
elif part == 'b':
    im.imsave('VasudevKhemkaCE4botsu.tif', binary_img_otsu)

print('otsu thresholding threshold:', T_otsu)
