# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 18:25:42 2018

@author: admin
"""

import cv2

img = cv2.imread('b1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

_, contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

i = 0
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.09 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
        cv2.imwrite('cropped\\' + str(i) + '_img.jpg', img)
        cv2.imshow("snip",img )

        i += 1