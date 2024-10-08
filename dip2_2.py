import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def func(mas_light,hist):
    shape = mas_light.shape
    result = np.zeros(shape,dtype=int)
    range1 = range(shape[0]-1)
    range2 = range(shape[1]-1)
    
    
    value = np.zeros(hist.shape)
        
    for i in range(255):
        for j in range(i):
            value[i] += hist[j]

    
    for i in range1:
        for j in range2:
                result[i][j] = value[mas_light[i][j]]
                
    return result
    
    
image1 = cv.imread('../images/lenna_bad.png')
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

channels = [0]
histSize = [256]
irange = [0, 256]

hist1 = cv.calcHist([gray_image1], channels, None, histSize, irange)

lut = lambda i: 255 * (func(i,hist1)/sum(hist1))
result_image = lut(gray_image1)

gs = plt.GridSpec(2, 2)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(gray_image1, cmap='gray')
plt.subplot(gs[1])
plt.imshow(result_image, cmap='gray')
plt.subplot(gs[2])
plt.hist(gray_image1.reshape(-1), 256, irange)
plt.subplot(gs[3])
plt.hist(result_image.reshape(-1), 256, irange)
plt.show()