# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:53:09 2017

@author: Pierre
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def getHesssianMatrix(img):
    """
    it returns the Hessian components A,B,C for each pixel
    """
    #The Sobel operator combines Gaussian smoothing and differentiation
    #cv2.CV_32F : to allow float numbers <0 and >255
    A = cv2.Sobel(img, cv2.CV_32F, dx=2,dy=0, ksize = windowSize)
    B = cv2.Sobel(img, cv2.CV_32F, dx=0, dy=2, ksize = windowSize)
    C = cv2.Sobel(img, cv2.CV_32F, dx=1,dy=1, ksize = windowSize)
    return A,B,C


def harrisCornerDetector(img, k):
    """
    img (greyscale): 2-dimension array, dtype=uint8
    k (Harris corner constant): float (usually 0.04-0.06)
    return the Harris response for each pixel
    """    
    A,B,C = getHesssianMatrix(img)
    
    #size of the input
    height, width = img.shape
    
    #calculate the response for each pixel: R = det(H) -k*Tr(H) where H is the Hessian matrix
    response = A*B - C**2 - k*(A+B)
    
    corners = [((i,j),response[j,i]) for j in range(height) for i in range(width)]
    dtype=[('location',[('xcorr',int),('ycorr',int)]),('response',float)]
    return np.array(corners, dtype=dtype)
    
def shiTomasiCornerDetector(img):
    """
    img (greyscale): 2-dimension array, dtype=uint8
    return the Harris response for each pixel
    """    
    A,B,C = getHesssianMatrix(img)
    
    #size of the input
    height, width = img.shape
    
    #calculate the response for each pixel:
    response = np.zeros((width,height))
    for i in range(width):
        for j in range(height):
            response[i,j] = np.min(np.linalg.eig(np.array([[A[i,j],C[i,j]],[C[i,j],B[i,j]]]))[0])
    
    corners = [((j,i),response[i,j]) for j in range(height) for i in range(width)]
    dtype=[('location',[('xcorr',int),('ycorr',int)]),('response',float)]
    return np.array(corners, dtype=dtype)

def triggsCornerDetector(img):
    """
    img (greyscale) : 2-dimension array, dtype=uint8
    return the Harris response for each pixel
    """
    A,B,C = getHesssianMatrix(img)
    #size of the input
    height, width = img.shape
    #calculate the response for each pixel :
    response = np.zeros((width,height))
    for i in range(width):
        for j in range(height):
            eigVal = np.linalg.eig(np.array([[A[i,j],C[i,j]],[C[i,j],B[i,j]]]))[0]
            response[i,j] = np.min(eigVal) - 0.05*np.max(eigVal)
    
    corners = [((j,i),response[i,j]) for j in range(height) for i in range(width)]
    dtype=[('location',[('xcorr',int),('ycorr',int)]),('response',float)]
    return np.array(corners, dtype=dtype)
    
def brownCornersDetection(img):
    """
    img (greyscale): 2-dimension array, dtype=uint8
    return the Harris response for each pixel
    """
    A,B,C = getHesssianMatrix(img)
    #size of the input
    height, width = img.shape
    #calculate the response for each pixel :
    response = np.zeros((width,height))
    for i in range(width):
        for j in range(height):
            eigVal = np.linalg.eig(np.array([[A[i,j],C[i,j]],[C[i,j],B[i,j]]]))[0]
            if (eigVal[0]+eigVal[1])==0:
                response[i,j] = 1 if eigVal[0]==eigVal[1] else -np.inf
            else:
                response[i,j] = eigVal[0]*eigVal[1]/(eigVal[0]+eigVal[1])
    corners = [((j,i),response[i,j]) for j in range(height) for i in range(width)]
    dtype=[('location',[('xcorr',int),('ycorr',int)]),('response',float)]
    return np.array(corners, dtype=dtype)

def getNBestCorners(corners, N) :
    """
    corners = [(location, response),...]
    """
    sortedTab = np.sort(corners, order='response')
    return sortedTab[len(sortedTab)-N:]
   

filename = 'pic/lena.jpg'
#filename = 'pic/cat.jpg'
#filename = 'pic/img_essai.png'

img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
k=0.06
windowSize = 3
numberOfCorners = 50


harrisCorners = harrisCornerDetector(img, k)
shiCorners = shiTomasiCornerDetector(img)
triggsCorners = triggsCornerDetector(img)

bestCornersHarris = getNBestCorners(harrisCorners, numberOfCorners)
bestCorners2 = getNBestCorners(shiCorners, numberOfCorners)
bestCorners3 = getNBestCorners(triggsCorners, numberOfCorners)


plt.figure()
markerSize = 10
markerWidth = 1.5
plt.plot(tuple(bestCornersHarris['location']['xcorr']), tuple(bestCornersHarris['location']['ycorr']), 'r+', markersize=markerSize, label='Harris Detector', markeredgewidth=markerWidth)
plt.plot(tuple(bestCorners2['location']['xcorr']), tuple(bestCorners2['location']['ycorr']), 'gx', markersize=markerSize, label='Shi and Tomasi Dectecor', markeredgewidth=markerWidth)
plt.plot(tuple(bestCorners3['location']['xcorr']), tuple(bestCorners3['location']['ycorr']), 'bo', markersize=5, label='Triggs Dectecor', markeredgewidth=markerWidth)

plt.legend()
plt.title(str(numberOfCorners) + ' best features')
plt.imshow(img, 'gray')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()