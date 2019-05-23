# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:05:44 2019

@author: GeePL
"""
import cv2
import numpy as np
from PIL import Image
import os
sep = os.sep

def his_gram(img_path, save_path, save=True):
    img_name = img_path.split(sep)[-1]
    image = cv2.imread(img_path, 0)    
    hist,bins = np.histogram(image.flatten(),256,[0,256]) 
    cdf = hist.cumsum() #计算累积直方图
    cdf_m = np.ma.masked_equal(cdf,0) #除去直方图中的0值
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())#等同于前面介绍的lut[i] = int(255.0 *p[i])公式
    cdf = np.ma.filled(cdf_m,0).astype('uint8') #将掩模处理掉的元素补为0
    result = cv2.LUT(image, cdf)
    if save:
        Image.fromarray(result).save(save_path+sep+img_name); 

if __name__=="__main__":
    # [65 66 67 70 75 77 79 80]
    print('start')
    data_path = r'D:\dataset2018\raw_img_with_xml'
    target_classes_list = ['65','66','67','70','79']
    for class_name in target_classes_list:
        img_dir_path = os.path.join(data_path, class_name)
        print(img_dir_path)
        img_names = [x for x in os.listdir(img_dir_path) if x[-4:]=='.jpg']
        for img_name in img_names:
            img_path = os.path.join(img_dir_path, img_name)
            print(img_path)
            his_gram(img_path, img_dir_path)
    print('end')
   