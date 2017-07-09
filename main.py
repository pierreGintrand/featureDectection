# -*- coding: utf-8 -*-
"""
@author: Pierre
"""


import matplotlib.pyplot as plt
import numpy as np
import cv2

import FeatureDetector



filename = 'pic/lena.jpg'
filename = 'pic/cat.jpg'
filename = 'pic/img_essai.png'



numberOfCorners = 50


lenaFeatures = FeatureDetector.FeatureDetector(filename, numberOfCorners)
lenaFeatures.getHarrisCorners()
lenaFeatures.getShiTomasiCorners()
lenaFeatures.getTriggsCorners()
lenaFeatures.getBrownCorners()

lenaFeatures.display()

#Display



