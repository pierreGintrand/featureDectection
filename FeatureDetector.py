# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


class FeatureDetector:
    """
    This class calculates features points thanks to differents methods
    """
    def __init__(self, fileName, numberOfCorners = 50, windowSize = 3):
        self.numberOfCorners = numberOfCorners
        self.windowSize = windowSize

        #open the image in grayscale mode
        self.img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

        #size of the image
        self.height, self.width = self.img.shape

        self.getHesssianMatrix()
        self.harrisCalculated = False
        self.shiTomasiCalculated = False
        self.triggsCalculated = False
        self.brownCalculated = False



    def getHesssianMatrix(self):
        """
        the function returns the Hessian components A,B,C for each pixel
        """
        #The Sobel operator combines Gaussian smoothing and differentiation
        #cv2.CV_32F : to allow float numbers <0 and >255
        self.A = cv2.Sobel(self.img, cv2.CV_32F, dx=2,dy=0, ksize = self.windowSize)
        self.B = cv2.Sobel(self.img, cv2.CV_32F, dx=0, dy=2, ksize = self.windowSize)
        self.C = cv2.Sobel(self.img, cv2.CV_32F, dx=1,dy=1, ksize = self.windowSize)

    def getHarrisCorners(self, k=0.06):
        """
        k (Harris corner constant): float (usually 0.04-0.06)
        it return the best features
        """    
        if not(self.harrisCalculated):
            #calculate the response for each pixel: R = det(H) -k*Tr(H) where H is the Hessian matrix
            response = self.A*self.B - self.C**2 - k*(self.A+self.B)

            corners = [((i,j),response[j,i]) for j in range(self.height) for i in range(self.width)]
            dtype=[('location',[('xcorr',int),('ycorr',int)]),('response',float)]
            harrisResponse = np.array(corners, dtype=dtype)
            self.harrisCorners = self.getNBestCorners(harrisResponse)
            self.harrisCalculated = True
        return self.harrisCorners


    def getShiTomasiCorners(self):
        if not(self.shiTomasiCalculated):
            #calculate the response for each pixel (minimu eigenvalue)
            response = np.zeros((self.width,self.height))
            for i in range(self.width):
                for j in range(self.height):
                    response[i,j] = np.min(np.linalg.eig(np.array([[self.A[i,j],self.C[i,j]],[self.C[i,j],self.B[i,j]]]))[0])
            
            corners = [((i,j),response[j,i]) for j in range(self.height) for i in range(self.width)]
            dtype=[('location',[('xcorr',int),('ycorr',int)]),('response',float)]
            shiTomasiResponse = np.array(corners, dtype=dtype)
            self.shiTomasiCorners = self.getNBestCorners(shiTomasiResponse)
            self.shiTomasiCalculated = True
        return self.shiTomasiCorners
        
        
    def getTriggsCorners(self, k=0.05):
        if not(self.triggsCalculated):
            #calculate the response for each pixel 
            response = np.zeros((self.width,self.height))
            for i in range(self.width):
                for j in range(self.height):
                    eigVal = np.linalg.eig(np.array([[self.A[i,j],self.C[i,j]],[self.C[i,j],self.B[i,j]]]))[0]
                    response[i,j] = np.min(eigVal) - k*np.max(eigVal)
            
            corners = [((i,j),response[j,i]) for j in range(self.height) for i in range(self.width)]
            dtype=[('location',[('xcorr',int),('ycorr',int)]),('response',float)]
            triggsResponse = np.array(corners, dtype=dtype)
            self.triggsCorners = self.getNBestCorners(triggsResponse)
            self.triggsCalculated = True
        return self.triggsCorners

    def getBrownCorners(self, k=0.05):
        if not(self.brownCalculated):
            #calculate the response for each pixel 
            response = np.zeros((self.width,self.height))
            for i in range(self.width):
                for j in range(self.height):
                    eigVal = np.linalg.eig(np.array([[self.A[i,j],self.C[i,j]],[self.C[i,j],self.B[i,j]]]))[0]
                    if (eigVal[0]+eigVal[1])==0:
                        response[i,j] = 1 if eigVal[0]==eigVal[1] else -np.inf
                    else:
                        response[i,j] = eigVal[0]*eigVal[1]/(eigVal[0]+eigVal[1])
            
            corners = [((i,j),response[j,i]) for j in range(self.height) for i in range(self.width)]
            dtype=[('location',[('xcorr',int),('ycorr',int)]),('response',float)]
            brownResponse = np.array(corners, dtype=dtype)
            self.brownCorners = self.getNBestCorners(brownResponse)
            self.brownCalculated = True
        return self.brownCorners
        
    
    def getNBestCorners(self, corners) :
        """
        corners = [(location, response),...]
        """
        sortedTab = np.sort(corners, order='response')
        return sortedTab[len(sortedTab)-self.numberOfCorners:]
        
        
    def display(self):
        plt.figure()
        markerSize = 10
        markerWidth = 1.5
        if(self.harrisCalculated):
            plt.plot(tuple(self.harrisCorners['location']['xcorr']), tuple(self.harrisCorners['location']['ycorr']), 'r+', markersize=markerSize, label='Harris Detector', markeredgewidth=markerWidth)
        if(self.shiTomasiCalculated):
            plt.plot(tuple(self.shiTomasiCorners['location']['xcorr']), tuple(self.shiTomasiCorners['location']['ycorr']), 'gx', markersize=markerSize, label='Shi Tomasi Detector', markeredgewidth=markerWidth)
        if(self.triggsCalculated):
            plt.plot(tuple(self.triggsCorners['location']['xcorr']), tuple(self.triggsCorners['location']['ycorr']), 'bo', markersize=markerSize/2, label='Triggs Detector', markeredgewidth=markerWidth)
        if(self.brownCalculated):
            plt.plot(tuple(self.brownCorners['location']['xcorr']), tuple(self.brownCorners['location']['ycorr']), 'w*', markersize=markerSize, label='Brown Detector', markeredgewidth=markerWidth)
        
        
        plt.legend()
        plt.title(str(self.numberOfCorners) + ' best features')
        plt.imshow(self.img, 'gray')
        plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
        plt.show()
    
        