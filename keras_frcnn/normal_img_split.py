# -*- coding: utf-8 -*-
"""
Created on Wed May  8 21:01:58 2019

@author: GeePL
"""

import cv2
import os
import numpy as np
from PIL import Image
from xml_create import xml_create
import random


path = '/home/data/xaj_guo/normal'
sep = os.sep
all_imgs = []
img_H = 3285
img_W = 1728
pieces_H = 4
pieces_W = 2
visualize = False

new_all_img = []
new_img_H = 900
new_img_W = 900
overlap_H = int((pieces_H*new_img_H - img_H)/(pieces_H-1))
overlap_W = int((pieces_W*new_img_W - img_W)/(pieces_W-1))

data_path = os.listdir(path)
imgs_path = [s for s in data_path if s[-4:]=='.jpg']
  
fpath_pngs = '/home/data/xaj_guo/900_normal/PNGImages'
if not os.path.exists(fpath_pngs):
    os.makedirs(fpath_pngs)    
fpath_xmls = '/home/data/xaj_guo/900_normal/Annotations'
if not os.path.exists(fpath_xmls):
    os.makedirs(fpath_xmls)
fpath_txts = '/home/data/xaj_guo/900_normal/ImageSet/Main'
if not os.path.exists(fpath_txts):
    os.makedirs(fpath_txts)

print('spliting')
for i in range(len(imgs_path)):
    try:       
        img_path = os.path.join(path, imgs_path[i])     
        raw_img = cv2.imread(img_path) 
        resized_img = cv2.resize(raw_img, (img_W, img_H))
        annotation_data = {'resized_img':resized_img,'width':img_W,
                           'height':img_H,'bboxes':[],'filepath':img_path}
        all_imgs.append(annotation_data)
    except Exception as e:
        print(e)
        continue
i = 0    
for img in all_imgs:
    try:
        fname = img['filepath'].split(sep)[-1][:-4]
        img_data = img['resized_img']
        i+=1
        print('{}/{}'.format(len(all_imgs),i))
        for h in range(pieces_H):
            for w in range(pieces_W):
                new_fname = fname+'_'+str(h)+str(w)+'.png'                              
                start_w = np.maximum(w*(new_img_W-overlap_W), 0)
                end_w = np.minimum(start_w+new_img_W, img_W)    
                start_h = np.maximum(h*(new_img_H-overlap_H), 0)
                end_h = np.minimum(start_h+new_img_H, img_H)
                
                if(img_data.shape[0]>=img_data.shape[1]):
                    new_img_data = img_data[start_h:end_h,start_w:end_w,:]
                else:
                    new_img_data = img_data[start_w:end_w,start_h:end_h,:]
              
                bboxes = img['bboxes']

                img_details = {'height':new_img_H, 'width':new_img_W,
                               'filename':new_fname[:-4],'bboxes':bboxes,
                               'img_data':img_data}
                xml_create(img_details,fpath_xmls)   
                Image.fromarray(new_img_data).save(fpath_pngs+sep+new_fname)          
            
                imageset = 'trainval' if random.randint(0,10) >= 2 else 'test'      
            
                with open(fpath_txts+sep+imageset+".txt","a+") as f:
                    f.write(new_fname[:-4]+"\n")
                            
    except Exception as e:
        print(e)
        print(h)
        print(w)
        continue
            
print('splting end')        