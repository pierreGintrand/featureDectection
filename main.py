# -*- coding: utf-8 -*-
"""
main.py
@author: Pierre
"""

import FeatureDetector

filename = 'pic/lena.jpg'

numberOfCorners = 50

lenaFeatures = FeatureDetector.FeatureDetector(filename, numberOfCorners)
lenaFeatures.getHarrisCorners()
lenaFeatures.getShiTomasiCorners()
lenaFeatures.getTriggsCorners()
lenaFeatures.getBrownCorners()

lenaFeatures.display()
