# -*- coding: utf-8 -*-
import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from xml_create import xml_create
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
def resize01(source_dir, target_classes, save, save_path='', 
           save_xml=False, visible=False, width=1800):
    all_resized_imgs = []
    classes_names =[x for x in os.listdir(source_dir) if x in target_classes]
    for class_name in classes_names:
        data_path = os.path.join(source_dir, class_name)
        jpg_names = [x for x in os.listdir(data_path) if x[-4:]=='.jpg']
        jpg_names.sort()
        xml_names = [x for x in os.listdir(data_path) if x[-4:]=='.xml']
        xml_names.sort()
        assert len(jpg_names) == len(xml_names)
        for i in range(len(jpg_names)):
            assert jpg_names[i][:-4] == xml_names[i][:-4]
            img_path = os.path.join(data_path, jpg_names[i])
            annot_path = os.path.join(data_path, xml_names[i])
            et = ET.parse(annot_path)
            element = et.getroot()
            element_objs = element.findall('object')
                    
            element_width = int(element.find('size').find('width').text)
            element_height = int(element.find('size').find('height').text)
            
            ratio = float(width)/element_width
            height = element_height * ratio
            
            ##读取图片，并压缩图片
            raw_img = cv2.imread(img_path) 
            resized_img = cv2.resize(raw_img, (width, height))
            
            if len(element_objs) > 0:
                resized_img_data = {'resized_img':resized_img,'width':width,
                                   'height':height,'bboxes':[],
                                   'filepath':img_path, 'filename':jpg_names[i][:-4]}
            
            for element_obj in element_objs:
                name = element_obj.find('name').text[:2]
                ## 读取标注框并压缩标注框
                obj_bbox = element_obj.find('bndbox')
                x1 = int(round(float(obj_bbox.find('xmin').text)*ratio))
                y1 = int(round(float(obj_bbox.find('ymin').text)*ratio))
                x2 = int(round(float(obj_bbox.find('xmax').text)*ratio))
                y2 = int(round(float(obj_bbox.find('ymax').text)*ratio))
                resized_img_data['bboxes'].append(
                        {'class':name,'x1':x1,'x2':x2,'y1':y1,'y2':y2})
                
                ## 是否在缩放后的图片中标出标注框
                if visible: 
                    cv2.rectangle(resized_img, (x1, y1),(x2, y2),(0,255,255),2) 
                    textOrg = (x1, y1)
                    cv2.putText(resized_img, name, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
                ## 是否保存缩放后的图片
                if save:
                    Image.fromarray(resized_img).save(save_path+sep+'resized_'+jpg_names[i]) 
                ## 是否保存缩放后的xml文件
                if save_xml:
                    xml_create(resized_img_data, save_path, file_name_prefix='resized_')
#                old_x1 = int(round(float(obj_bbox.find('xmin').text)))
#                old_y1 = int(round(float(obj_bbox.find('ymin').text)))
#                old_x2 = int(round(float(obj_bbox.find('xmax').text)))
#                old_y2 = int(round(float(obj_bbox.find('ymax').text)))
#                cv2.rectangle(raw_img, (old_x1, old_y1),(old_x2, old_y2),(0,255,255),2)          
#                Image.fromarray(raw_img).save(fpath_tmp+'/'+imgs_path[i][:-4]+'.jpg')               
            all_resized_imgs.append(resized_img_data)
    return all_resized_imgs

"""
source_dir
    包含jpg和xml文件的文件夹
    未作瑕疵分类的文件
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
def resize02(source_dir, save, save_path='', 
           save_xml=False, visible=False, width=1800):
    all_resized_imgs = []
    classes_count = {}
    class_mapping = {}
    data_path = os.path.join(source_dir)
    jpg_names = [x for x in os.listdir(data_path) if x[-4:]=='.jpg']
    jpg_names.sort()
    xml_names = [x for x in os.listdir(data_path) if x[-4:]=='.xml']
    xml_names.sort()
    assert len(jpg_names) == len(xml_names)
    for i in range(len(jpg_names)):
        assert jpg_names[i][:-4] == xml_names[i][:-4]
        img_path = os.path.join(data_path, jpg_names[i])
        annot_path = os.path.join(data_path, xml_names[i])
        et = ET.parse(annot_path)
        element = et.getroot()
        element_objs = element.findall('object')
                
        element_width = int(element.find('size').find('width').text)
        element_height = int(element.find('size').find('height').text)
        
        ratio = float(width)/element_width
        height = int(element_height * ratio)
        
        ##读取图片，并压缩图片
        raw_img = cv2.imread(img_path) 
        resized_img = cv2.resize(raw_img, (width, height))
        
        if len(element_objs) > 0:
            resized_img_data = {'resized_img':resized_img,'width':width,
                               'height':height,'bboxes':[],
                               'filepath':img_path, 'filename':jpg_names[i][:-4]}
        
        for element_obj in element_objs:
            class_name = element_obj.find('name').text[:2]
            assert len(class_name)==2
            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1
            if class_name not in class_mapping:
                class_mapping[class_name] = len(class_mapping)
            ## 读取标注框并压缩标注框
            obj_bbox = element_obj.find('bndbox')
            x1 = int(round(float(obj_bbox.find('xmin').text)*ratio))
            y1 = int(round(float(obj_bbox.find('ymin').text)*ratio))
            x2 = int(round(float(obj_bbox.find('xmax').text)*ratio))
            y2 = int(round(float(obj_bbox.find('ymax').text)*ratio))
            resized_img_data['bboxes'].append(
                    {'class':class_name,'x1':x1,'x2':x2,'y1':y1,'y2':y2})
            
            ## 是否在缩放后的图片中标出标注框
            if visible: 
                cv2.rectangle(resized_img, (x1, y1),(x2, y2),(0,255,255),2) 
                textOrg = (x1, y1)
                cv2.putText(resized_img, class_name, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
            ## 是否保存缩放后的图片
            if save:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                Image.fromarray(resized_img).save(save_path+sep+'resized_'+jpg_names[i]) 
            ## 是否保存缩放后的xml文件
            if save_xml:
                xml_create(resized_img_data, save_path, file_name_prefix='resized_')              
        all_resized_imgs.append(resized_img_data)
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
def split(all_resized_imgs, save, save_path, height_slices, width_slices,
           max_pieces=6, offset=30, height=900, width=900,
           visible=False, save_xml=True):
    for resized_img in all_resized_imgs:
        splited_imgs_per_resized_img = []
        #overstep_jpgs = []
        resized_img_name = resized_img['filepath'].split(sep)[-1][:-4]
        resized_img_data = resized_img['resized_img']
        resized_img_height = resized_img['height']
        resized_img_width = resized_img['width']
        height_overlap = int((height_slices*height - resized_img_height)/np.maximum(height_slices-1, 1))
        width_overlap = int((width_slices*width - resized_img_width)/np.maximum(width_slices-1, 1))
        for h in range(height_slices):
            for w in range(width_slices):
                splited_resized_img_name = resized_img_name+'_'+str(h)+'_'+str(w)+'.png'
                start_w = np.maximum(w*(width-width_overlap), 0)
                end_w = np.minimum(start_w+width, resized_img_width)    
                start_h = np.maximum(h*(height-height_overlap), 0)
                end_h = np.minimum(start_h+height, resized_img_height)
                
                if(resized_img_data.shape[0]>=resized_img_data.shape[1]):
                    splited_resized_img_data = resized_img_data[start_h:end_h,start_w:end_w,:]
                else:
                    splited_resized_img_data = resized_img_data[start_w:end_w,start_h:end_h,:]
                
                iou_between_01 = False
                iou_equals_1 = 0
                bboxes = resized_img['bboxes']
                if len(bboxes)>0:
                    new_bboxes = []
                    for bbox in bboxes:
                        x1 = bbox['x1']
                        y1 = bbox['y1']
                        x2 = bbox['x2']
                        y2 = bbox['y2']
                        if y1<start_h and y2>end_h:
                            y1 = start_h+offset
                            y2 = end_h - offset
                            #overstep_jpgs.append(splited_resized_img_name)
                        if x1 <start_w and x2>end_w:
                            x1 = start_w+offset
                            x2 = end_w-offset
                            #overstep_jpgs.appen(splited_resized_img_name)   
                        iou = iou_upon_bbox([start_w, start_h, end_w, end_h],
                                                  [x1,y1,x2,y2])
                        
                        if iou == 0:
                            pass
                        elif 0<iou<0.99:
                            iou_between_01 = True 
                            break
                        else:
                            new_x1 = x1 - start_w
                            new_y1 = y1 - start_h
                            new_x2 = x2 - start_w
                            new_y2 = y2 - start_h
                            new_bboxes.append({'x1':new_x1, 'x2':new_x2,
                                               'y1':new_y1, 'y2':new_y2,
                                               'class':bbox['class']})
                            iou_equals_1 += 1
                                              
                    if(iou_between_01):
                        pass
                    else:
                        if iou_equals_1 != 0:
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
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                Image.fromarray(splited_img_data).save(save_path+sep+'splited_'+filename+'.png') 
            ## 是否保存裁剪后的xml文件
            if save_xml:
                xml_create(img_details, save_path, file_name_prefix='splited_')               

if __name__=="__main__":
    for num in [69,70,74,78,79]:
        data_path = r'/opt/xdata/chaoy/dataset2018/'+str(num)
        all_imgs = resize02(data_path, save=True, 
                       save_path=r'/home/guopl/resized/'+str(num), 
                       save_xml=False, visible=True, width=1800)
    