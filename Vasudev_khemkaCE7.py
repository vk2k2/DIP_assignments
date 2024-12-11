import imageio.v2 as im
import numpy as np
import matplotlib.pyplot as plt



img = im.imread("patterns.tif")
F_img = np.fft.fft2(img)
Fimgshift = np.fft.fftshift(F_img)
(M,N) = img.shape
Do = M/60
Gimshift = np.zeros([M,N])
H = np.zeros([M,N])
n = 1
for u in range(M):
    for v in range(N):
        H[u,v] = (1/(1+(((u - M/2)**2+(v-N/2)**2)/Do**2)**n))

Gimshift = np.multiply(Fimgshift,H)
     
Gimg = np.fft.ifftshift(Gimshift)
gimg = abs(np.fft.ifft2(Gimg))
gimg = (255*gimg/gimg.max()).astype(np.uint8)

fig1 = plt.figure(figsize=(7, 10))

fig1.add_subplot(3, 1, 1)
plt.axis('off')
plt.imshow(img)

fig1.add_subplot(3, 1,2)
plt.axis('off')

Fimgplot = np.log10(1+abs(Gimshift))
Fimgplot = (255*Fimgplot/Fimgplot.max()).astype(np.uint8)

plt.imshow(Fimgplot)

fig1.add_subplot(3, 1, 3)
plt.axis('off')
plt.imshow(gimg)
plt.savefig('VasudevKhemkaCE7a.tif')


Jimshift = np.multiply(Fimgshift,(1-H))
Jimg = np.fft.ifftshift(Jimshift)
jimg = abs(np.fft.ifft2(Jimg))
jimg = (255*jimg/jimg.max()).astype(np.uint8)

fig2 = plt.figure(figsize=(7, 10))

fig2.add_subplot(3, 1, 1)
plt.axis('off')
plt.imshow(img)

fig2.add_subplot(3, 1,2)
plt.axis('off')

Fimgplot2 = np.log10(1+abs(Jimshift))
Fimgplot2 = (255*Fimgplot2/Fimgplot2.max()).astype(np.uint8)

plt.imshow(Fimgplot2)

fig2.add_subplot(3, 1, 3)
plt.axis('off')
plt.imshow(jimg)
plt.savefig('VasudevKhemkaCE7b.tif')


