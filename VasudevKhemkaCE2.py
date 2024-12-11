import imageio.v2 as im
import numpy as np
from matplotlib import pyplot as plt

def img_filter(image, X):
    m,n=image.shape
    l,b = X.shape
    lower_l = int(np.floor(l/2))
    upper_l = int(np.ceil(l/2))
    lower_b = int(np.floor(b/2))
    upper_b = int(np.ceil(b/2))
    filtered_img = np.zeros((m,n))
    for i in range(lower_l,m-lower_l):
        for j in range(lower_b,n-lower_b):
            arr = image[i-lower_l:i+upper_l,j-lower_b:j+upper_b] 
            filtered_img[i][j] = np.sum(np.multiply(arr,X))
            
    return filtered_img

def CE2():
    part = input()
    if part == 'a':
        funca()
    elif part=='b':
        funcb()
    elif part =='c':
        func()
    else : print(' enter a,b or c')


def funca():

    img=im.imread('einstein.tif')
    a=[0]*256
    b=[0]*256
    height,width=img.shape
  #histogram generation
    for i in range(width):
        for j in range(height):
            g=img[j,i]
            a[g]=a[g]+1

    ab= list(range(0,256))
    plt.bar(ab,a)
    plt.show()
  #equilization
  
    for g in range(1,256):
        a[g]=a[g]/(height*width)
    b[0]=a[0]*256
    for g in range(1,256):
        b[g]=b[g-1]+(a[g]*256)
    plt.bar(ab,b)
    plt.show()

    for i in range(0,width):
        for j in range(0,height):
            g=img[j,i]
            img[j,i]=b[g]

    im.imsave('VasudevKhemkaCE2a.tif',img)

def funcb():
    img = im.imread('instagram.tif')
    m,n=img.shape
    img1=im.imread('instagram.tif')
    img2=im.imread('instagram.tif')
    img3=im.imread('instagram.tif')
#3x3
    kernel_3x3 = np.ones((3,3))/9
    img1 = img_filter(img, kernel_3x3)

#comment out the rest if it takes too long to run...

#5x5    
    kernel_5x5 = np.ones((5,5))/25
    img2 = img_filter(img, kernel_5x5)

#7x7    
    kernel_7x7 = np.ones((7,7))/49
    img3 = img_filter(img, kernel_7x7)
    
    im.imsave('VasudevKhemkaCE2b3x3.tif',img1)
 
    print('filter size 3x3 is most appropriate, as patches of the same \
           grayscale values are less noticable, compared to 5x5 or 7x7.')

def func():

    img = im.imread('moon.tif')
    m,n=img.shape
    img1=im.imread('moon.tif')


    def highboost(img,blur):
        k=float(1.3)
        highboosted = (k+1)*img - blur*k
        
        #normalize higher values
        highboosted = 255*highboosted/highboosted.max()
        
        #values lower than 0 become 0
        for i in range(1,m):
            for j in range(1,n):
                if highboosted[i][j]<0: highboosted[i][j]=0
                
        #float -> 8 bit unsigned int
        highboosted = highboosted.astype(np.uint8)
        return highboosted

    #3x3  
    kernel_3x3 = np.ones((3,3))/9
    img1 = img_filter(img, kernel_3x3)

    highboost1 = highboost(img,img1) 


        
    
    print('best image generated with 3x3 smoothing filter /n')

    im.imsave('highboost1.tif', highboost1)
    
    print('Value of k used: 2.5. Above this value, contrast seems to decrease\
          without much increase in sharpness.')

    
    
CE2()
