import imageio.v2 as im
import numpy as np
import matplotlib.pyplot as plt

img_rgb=im.imread("colors.tif")
x = img_rgb.shape

c = 255
gamma = 2
gamma_transformed_img_rgb = c * np.power((img_rgb/255),gamma)
gamma_transformed_img_rgb = gamma_transformed_img_rgb.astype(np.uint8)

img_rgb_n = img_rgb/c
img_hsi = np.zeros(shape=x)

for (r,g,b,H,S,I) in np.nditer((img_rgb_n[:,:,0],img_rgb_n[:,:,1],img_rgb_n[:,:,2],img_hsi[:,:,0],img_hsi[:,:,1],img_hsi[:,:,2]), op_flags=['readwrite']):
    m = min(r,min(g,b)) 
    if b==g==r:
        H[...] = 0
    elif b <=g:
        H[...] = 180/np.pi*np.arccos(0.5 * ((r - g) + (r - b)) / np.sqrt(((r - g) ** 2) + (r - b) * (g - b)))
    else:
        H[...] = 360 - 180/np.pi*np.arccos(0.5 * ((r - g) + (r - b)) / np.sqrt(((r - g) ** 2) + (r - b) * (g - b)))
    
    I[...] = (r+g+b)/3
    
    if I==0:
        S[...] = 0
    else:
        S[...] = (1-m/I)
        
gamma_hsi = 1.8
gamma_transformed_img_hsi = np.copy(img_hsi)
gamma_transformed_img_hsi[:,:,2] = gamma_transformed_img_hsi[:,:,2]
gamma_transformed_img_hsi[:,:,2] = np.power(gamma_transformed_img_hsi[:,:,2],gamma_hsi)/gamma_transformed_img_hsi[:,:,2].max()
img_rgb_conv = np.zeros(shape=x)
for (r,g,b,h,S,I) in np.nditer((img_rgb_conv[:,:,0],img_rgb_conv[:,:,1],img_rgb_conv[:,:,2],gamma_transformed_img_hsi[:,:,0],gamma_transformed_img_hsi[:,:,1],gamma_transformed_img_hsi[:,:,2]), op_flags=['readwrite']):
        
    if 0 <= h <= 120 :
        b[...] = I * (1 - S)
        r[...] = I * (1 + (S * np.cos(h*np.pi/180) / np.cos(60*np.pi/180 - h*np.pi/180)))
        g[...] = I * 3 - (r + b)
    elif 120 < h <= 240:
        
        r[...] = I * (1 - S)
        g[...] = I * (1 + (S * np.cos((h-120)*np.pi/180) / np.cos(60*np.pi/180 - (h-120)*np.pi/180)))
        b[...] = 3 * I - (r + g)
    elif 0 < h <= 360:
        
        g[...] = I * (1 - S)
        b[...] = I * (1 + (S * np.cos((h-240)*np.pi/180) / np.cos(60*np.pi/180 - (h-240)*np.pi/180)))
        r[...] = I * 3 - (g + b)    
        
img_rgb_conv = (255*img_rgb_conv).astype(np.uint8)


fig = plt.figure(figsize=(7, 10))

fig.add_subplot(2, 1, 1)
plt.imshow(gamma_transformed_img_rgb)
plt.axis('off')
plt.title("gamma_transformed_img_rgb")

fig.add_subplot(2, 1, 2)
plt.imshow(img_rgb_conv)
plt.axis('off')
plt.title("gamma_transformed_img_hsi")
plt.savefig('VasudevKhemkaCE6ab.tif')

img_pink = np.zeros((x[0],x[1]))
for (H,S,I,clas) in np.nditer((img_hsi[:,:,0],img_hsi[:,:,1],img_hsi[:,:,2],img_pink),op_flags=['readwrite']):
    if 320<H<340:
        if S>0.1:
            clas[...] = 255
            
im.imsave("VasudevKhemkaCE6c.tif",img_pink.astype(np.uint8))
    #H: 320 to 340 pink
    
    

    