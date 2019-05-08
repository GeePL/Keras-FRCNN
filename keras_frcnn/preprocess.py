# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:15:38 2018

@author: GeePL
"""

import cv2
import numpy as np

def build_filters():
    filters = []
    ksize = 10 # gabor尺度
    lamda = np.pi/2.0 #波长
    
#    for theta in np.arange(0,np.pi,np.pi/4): #gabor方向，0°，45°，90°，135°，共四个
#        for K in range(len(ksize)): 
#            kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
#            kern /= 1.5*kern.sum()
#            filters.append(kern)
    theta = np.pi/18
    kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
    kern /= 1.5*kern.sum()
    filters.append(kern)
    return filters

 ###    Gabor滤波过程
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


def getGabor(img,filters):
    res = [] #滤波结果
    for i in range(len(filters)):        
        res1 = process(img, filters[i])
        res.append(np.asarray(res1))
        cv2.imwrite('./gabor_'+str(i)+'.png',res1)
    return res  #返回滤波结果

def his_gram(img_path):
    image = cv2.imread(img_path, 0)    
    hist,bins = np.histogram(image.flatten(),256,[0,256]) 
    cdf = hist.cumsum() #计算累积直方图
    cdf_m = np.ma.masked_equal(cdf,0) #除去直方图中的0值
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())#等同于前面介绍的lut[i] = int(255.0 *p[i])公式
    cdf = np.ma.filled(cdf_m,0).astype('uint8') #将掩模处理掉的元素补为0

    result = cv2.LUT(image, cdf)
    
    cv2.imwrite("opencv_lut.jpg", result)
    
def gamma(img_path):
    img = cv2.imread(img_path,0)
    img = np.power(img/255.0, 1.2)
    cv2.imwrite("gamma_transfer.jpg",img)

def canny_transfer(img_path):
    img = cv2.imread(img_path)
    canny = cv2.Canny(img, 10,200)
    cv2.imwrite("canny.jpg",canny)  

if __name__=='__main__':
#    filters = build_filters()
#    img = cv2.imread('./opencv_lut.jpg')
#    res = getGabor(img,filters)
    his_gram('./81_20171228_193903_11482.png')
#    gamma('./opencv_lut.jpg')
    canny_transfer('./opencv_lut.jpg')