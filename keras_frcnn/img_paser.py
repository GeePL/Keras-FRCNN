# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:12:43 2019

@author: GeePL
"""
import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from xml_create import xml_create
from data_augment import augment
import random
sep = os.sep

#左下-右上
def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h

#交并比
def iou_upon_bbox(a, b):
	# a is the 900*900
    # b is the gt bbox

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0
    
	area_i = intersection(a, b)
	area_u = (b[2] - b[0]) * (b[3] - b[1])

	return float(area_i) / float(area_u + 1e-6)

"""
source_dir
    包含所有瑕疵类别的目录，根目录
    要求瑕疵图片已经按编号分类好
target_classes
    希望被执行操作的瑕疵类别
width
    图片缩放后的期望宽度
save,save_path
    是否要保存缩放后的图片以及对应的存放路径
    要求路径名具体到瑕疵编号
visible
    缩放后的标注框是否显示在缩放后的图片上
width
    默认的缩放后的图片的宽度
    
output
    一个字典，每个entry包含：
        缩放后的图片
        图片高与宽
        图片路径
        标注框
            瑕疵类别
            瑕疵坐标：左下-右上
"""
def resize(source_dir, target_classes, save, save_path='', 
           save_xml=False, visible=False, width=900):
    all_resized_imgs = []
    classes_names =[x for x in os.listdir(source_dir) if x in target_classes]
    for class_name in classes_names:
        data_path = os.path.join(source_dir, class_name)
        jpg_names = [x for x in os.listdir(data_path) if x[-4:]=='.jpg']
        jpg_names.sort()
        xml_names = [x for x in os.listdir(data_path) if x[-4:]=='.xml']
        xml_names.sort()
        assert len(jpg_names) == len(xml_names)
        print(len(jpg_names))
        for i in range(len(jpg_names)):
            assert jpg_names[i][:-4] == xml_names[i][:-4]
            img_path = os.path.join(data_path, jpg_names[i])
            annot_path = os.path.join(data_path, xml_names[i])
            et = ET.parse(annot_path)
            element = et.getroot()
            element_objs = element.findall('object')
            
            raw_img = cv2.imread(img_path)         
            element_width = raw_img.shape[1];
            element_height = raw_img.shape[0];
            
            width_ratio = float(width)/element_width
            height_ratio = width_ratio / 1.5
            height =int(element_height * height_ratio)
            
            ##读取图片，并压缩图片
            resized_img = cv2.resize(raw_img, (width, height))
            
            if len(element_objs) > 0:
                resized_img_data = {'resized_img':resized_img,'width':width,
                                   'height':height,'bboxes':[],
                                   'filepath':img_path, 'filename':jpg_names[i][:-4]}
            
                for element_obj in element_objs:
                    name = element_obj.find('name').text[:2]
                    ## 读取标注框并压缩标注框
                    obj_bbox = element_obj.find('bndbox')
                    x1 = int(round(float(obj_bbox.find('xmin').text)*width_ratio))
                    y1 = int(round(float(obj_bbox.find('ymin').text)*height_ratio))
                    x2 = int(round(float(obj_bbox.find('xmax').text)*width_ratio))
                    y2 = int(round(float(obj_bbox.find('ymax').text)*height_ratio))
                    resized_img_data['bboxes'].append(
                            {'class':name,'x1':x1,'x2':x2,'y1':y1,'y2':y2})
                    
                    ## 是否在缩放后的图片中标出标注框
                    if visible: 
                        cv2.rectangle(resized_img, (x1, y1),(x2, y2),(0,255,255),2) 
                        textOrg = (x1, y1)
                        cv2.putText(resized_img, name, 
                                    textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
                    ## 是否保存缩放后的图片
                    if save:
                        save_path_tmp = save_path+sep+class_name
                        if not os.path.exists(save_path_tmp):
                            os.makedirs(save_path_tmp)
                        Image.fromarray(resized_img) \
                        .save(save_path_tmp+sep+'resized_'+jpg_names[i]) 
                    ## 是否保存缩放后的xml文件
                    if save_xml:
                        xml_create(resized_img_data, save_path_tmp, 
                                   file_name_prefix='resized_')
            all_resized_imgs.append(resized_img_data)
    print("resized done")
    return all_resized_imgs
"""
all_resized_imgs
    一个List， 包含所有缩放后的图片
save_path
    保存裁剪后的png图片和对应xml文件的位置，要求路径具体到瑕疵编号
height_slices，width_slices
    希望切割的片数
max_pieces
    从一张缩放后的图片获得的最大的切割的数目
offset
    当标注框纵/横向横跨整个splied_img时，需要的偏移量
height=900, width=900
    默认的期望的切割后的图片大小
"""
def split(all_resized_imgs, target_class,save, save_path, 
          height_slices, width_slices,
          max_pieces=6, offset=30, height=900, width=900,
          visible=False, save_xml=True):
    print("split begin")
    all_splited_imgs = []
    for resized_img in all_resized_imgs:
        splited_imgs_per_resized_img = []
        #overstep_jpgs = []
        resized_img_name = resized_img['filepath'].split(sep)[-1][:-4]
        resized_img_data = resized_img['resized_img']
        resized_img_height = resized_img['height']
        resized_img_width = resized_img['width']
        bboxes = resized_img['bboxes']
        height_overlap = int((height_slices*height - resized_img_height)/np.maximum(height_slices-1, 1))
        width_overlap = int((width_slices*width - resized_img_width)/np.maximum(width_slices-1, 1))
        for h in range(height_slices):
            for w in range(width_slices):
                splited_resized_img_name = resized_img_name+'_'+str(h)+'_'+str(w)+'.jpg'
                start_w = np.maximum(w*(width-width_overlap), 0)
                end_w = np.minimum(start_w+width, resized_img_width)    
                start_h = np.maximum(h*(height-height_overlap), 0)
                end_h = np.minimum(start_h+height, resized_img_height)
                
                if(resized_img_data.shape[0]>=resized_img_data.shape[1]):
                    splited_resized_img_data = resized_img_data[start_h:end_h,start_w:end_w,:]
                else:
                    splited_resized_img_data = resized_img_data[start_w:end_w,start_h:end_h,:]
                
                if len(bboxes)>0:
                    new_bboxes = []
                    for bbox in bboxes:
                        x1 = bbox['x1']
                        y1 = bbox['y1']
                        x2 = bbox['x2']
                        y2 = bbox['y2']
                        """
                        ## 因为图片的宽度已经是最小的了，所以
                        ## 不会出现标注框的宽度比图片的宽度大的情况
                        ## 左右贯穿
                        if x1 <start_w and x2>end_w:
                            x1 = start_w+offset
                            x2 = end_w-offset  
                        ## 局部相交
                        ## 只需要考虑在高度方向上，分片只包含了标注框的局部
                        ## 而且标注框还不是贯穿图片的情况
                        """
                        ## 上下贯穿
                        if y1<start_h and y2>end_h:
                            y1 = start_h+offset
                            y2 = end_h - offset
                        ## 局部相交
                        elif y1<start_h and y2>start_h and y2<end_h:
                            y1 = start_h+offset
                        elif y1>start_h and y1<end_h and y2>end_h:
                            y2 = end_h-offset
                        
                        iou = iou_upon_bbox([start_w, start_h, end_w, end_h],
                                                  [x1,y1,x2,y2])
                        
                        if iou < 0.7:
                            continue
                        else:
                            new_x1 = x1 - start_w
                            new_y1 = y1 - start_h
                            new_x2 = x2 - start_w
                            new_y2 = y2 - start_h
                            new_bboxes.append({'x1':new_x1, 'x2':new_x2,
                                               'y1':new_y1, 'y2':new_y2,
                                               'class':bbox['class']})                                            
                    if(len(new_bboxes)>0):
                        img_details = {'height':height, 'width':width,
                                       'filename':splited_resized_img_name[:-4],
                                       'bboxes':new_bboxes,
                                       'new_img_data':splited_resized_img_data}
                        splited_imgs_per_resized_img.append(img_details)
        if len(splited_imgs_per_resized_img)>max_pieces:
            random.shuffle(splited_imgs_per_resized_img)
            splited_imgs_per_resized_img = splited_imgs_per_resized_img[:max_pieces]
        for img_details in splited_imgs_per_resized_img:
            filename = img_details['filename']       
            splited_img_data = img_details['new_img_data']     
            ## 是否在裁剪后的图片中标出标注框
            if visible: 
                for bbox in img_details['bboxes']:
                    class_name = bbox['class']
                    x1 = bbox['x1']
                    y1 = bbox['y1']
                    x2 = bbox['x2']
                    y2 = bbox['y2']
                    cv2.rectangle(splited_img_data, (x1, y1),(x2, y2),(0,255,255),2) 
                    textOrg = (x1, y1)
                    cv2.putText(splited_img_data, class_name, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
            ## 是否保存裁剪后的图片
            if save:
                for bbox in img_details['bboxes']:
                    class_name = bbox['class']
                    if class_name not in target_class:
                        continue
                    save_path_tmp = save_path+sep+class_name
                    if not os.path.exists(save_path_tmp):
                        os.makedirs(save_path_tmp)
                    Image.fromarray(splited_img_data).save(save_path_tmp+sep+'splited_'+filename+'.jpg') 
                    ## 是否保存裁剪后的xml文件
                    if save_xml:
                        xml_create(img_details, save_path_tmp, file_name_prefix='splited_')   
        all_splited_imgs.append(splited_imgs_per_resized_img)
    print("split done")
    return all_splited_imgs

def get_flaw_data(source_dir, target_class, width=900):
    all_flaw_imgs = []
    for class_name in target_class:
        data_path = os.path.join(source_dir, class_name)
        jpg_names = [x for x in os.listdir(data_path) if x[-4:]=='.jpg']
        jpg_names.sort()
        xml_names = [x for x in os.listdir(data_path) if x[-4:]=='.xml']
        xml_names.sort()
        assert len(jpg_names) == len(xml_names)
        print(len(jpg_names))
        for i in range(len(jpg_names)):
            img_path = os.path.join(data_path, jpg_names[i])
            annot_path = os.path.join(data_path, xml_names[i])
            et = ET.parse(annot_path)
            element = et.getroot()
            element_objs = element.findall('object')
            annotation_data = {'filepath': img_path, 
                               'width':width ,'height': width, 'bboxes': []}
            
            if np.random.randint(0,10) < 7:
                annotation_data['imageset'] = 'trainval'
            else:
                annotation_data['imageset'] = 'test'

            if len(element_objs) > 0:
                for element_obj in element_objs:
                    name = element_obj.find('name').text[:2]
                    obj_bbox = element_obj.find('bndbox')
                    x1 = int(round(float(obj_bbox.find('xmin').text)))
                    y1 = int(round(float(obj_bbox.find('ymin').text)))
                    x2 = int(round(float(obj_bbox.find('xmax').text)))
                    y2 = int(round(float(obj_bbox.find('ymax').text)))
                    annotation_data['bboxes'].append({
                            'class': name, 'x1': x1, 'x2': x2, 
                            'y1': y1, 'y2': y2, 'difficult': 1})
            all_flaw_imgs.append(annotation_data)
    return all_flaw_imgs

def get_normal_data():
    pass

  
if __name__=='__main__':
    #target_classes=['65','66','67','70', '75','77','79','80']
#    all_resized_imgs = resize(source_dir=r'D:\dataset2018-05-23\raw_img_with_histogram',
#                        target_classes=['65','66','67','70','75','77','79','80'], save=False, 
#                        save_path=r'D:\dataset2018-05-23\resized_img_with_histogram', 
#                        save_xml=False, visible=False, width=900)
#    all_splited_imgs = split(all_resized_imgs=all_resized_imgs, 
#                             target_class=['65','66','67','70', '75','77','79','80'],
#                             save=True, 
#                             save_path=r'D:\dataset2018-05-23\splited_img_with_histogram', 
#                             height_slices=6, width_slices=1,
#                             max_pieces=3, offset=5, height=900, width=900,
#                             visible=False, save_xml=True)
    all_flaw_imgs = get_flaw_data(source_dir=r'D:\dataset2018-05-23\splited_img_with_histogram',
                                  target_class=['65','66','67','70','75','77','79','80'])
    count=0
    print('*******')
    for flaw_img in all_flaw_imgs:
        if flaw_img['imageset']=='trainval':
            count+=1
    print(count)
    