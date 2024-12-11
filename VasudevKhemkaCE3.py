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
    for i in range(lower_l,m-lower_l): # make out of edges = 0
        for j in range(lower_b,n-lower_b):
            arr = image[i-lower_l:i+upper_l,j-lower_b:j+upper_b] 
            filtered_img[i][j] = np.sum(np.multiply(arr,X))
            
    return filtered_img
    
def CE3():

    img = im.imread('skeleton.tif')
    m,n=img.shape
    
# gaussian smoothing
    var = 1
    k = np.power(np.e,0.5/var)                            # variance = 1
    G = np.array([[1,k,1],[k,np.power(k,2),k],[1,k,1]])
    w_sum = np.sum(G)
    G = G/w_sum
    
    smooth_img = img_filter(img, G)
    
            
# Laplacian 
    L = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    
    laplac_img = img_filter(smooth_img, L)


# Sharpening
    sharp_img = img - 2*laplac_img

# plotting and normalizing  

    sharp_img_norm = sharp_img + np.abs(sharp_img.min())
    sharp_img_norm = 255*sharp_img_norm/sharp_img_norm.max()
    sharp_img_norm = sharp_img_norm.astype(np.uint8)

    smooth_img = smooth_img.astype(np.uint8)
    
    laplac_img = laplac_img + np.abs(laplac_img.min())
    laplac_img = laplac_img.astype(np.uint8)
        
    fig = plt.figure(figsize=(7, 10))
    
    fig.add_subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("original")
    
    fig.add_subplot(2, 2, 2)
    plt.imshow(smooth_img)
    plt.axis('off')
    plt.title("smoothened image")
    
    fig.add_subplot(2, 2, 3)
    plt.imshow(laplac_img)
    plt.axis('off')
    plt.title("Laplacian image")
    
    fig.add_subplot(2, 2, 4)
    plt.imshow(sharp_img_norm)
    plt.axis('off')
    plt.title("sharpened image")
    
    plt.savefig('VasudevKhemkaCE3a.png')
    
# gamma transformation on sharpened image
    
    c = 255
    gamma = 0.9
    gamma_transformed_img = c * np.power((sharp_img_norm/255),gamma)
    gamma_transformed_img = gamma_transformed_img.astype(np.uint8)
    print(' gamma value is: ', gamma)
    im.imsave("VasudevKhemkaCE3b.tif", gamma_transformed_img)
    
CE3()