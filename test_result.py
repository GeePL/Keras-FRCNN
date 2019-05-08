# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:42:18 2018

@author: GeePL
"""
from keras_frcnn.pascal_voc_parser import get_data
from tool import frcnn
detector = frcnn(model_path='./model_frcnn.hdf5',config_filename='./config.pickle')
pre_vol, real_vol, acc_vol = detector.detectSaveByXml(voc_path='./cy_image',save_all=True, save_failed=True,
                                                      save_path='./results_imgs',img_set='test',zero_path=None,bbox_threshold=0.0)
ignoreClass=['65','66','68','69','70','78','79','80','65','74','76']
tp = detector.getResultFromRealPre(pre=pre_vol, real=real_vol, ignoreClass=ignoreClass)

#all_imgs, classes_count, class_mapping = get_data('./cy_image')
#val_imgs = [s for s in all_imgs if s['imageset'] == 'test']
#val_whole = []
#preflaws = []
#realflaws = []
#
#for val_img in val_imgs:
#    filepath = val_img['filepath']
#    filename = filepath.split('/')[-1]
#    filename = filename[:filename.rfind('_')]
#    if not filename in val_whole:
#        val_whole.append(filename)
