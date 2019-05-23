# -*- coding: utf-8 -*-
import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from xml_create import xml_create
import random
import shutil
import config
from data_augment import augment
sep = os.sep
C = config.Config()
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
<<<<<<< HEAD:keras_frcnn/utils.py
             white_list=['61','62','63','64','71','72','73'],
=======
           white_list=['61','62','63','64','71','72','73'],
>>>>>>> 8c8f50e3f0157b68f9105770f94b75ff7a65f348:keras_frcnn/flaw_img_split.py
           save_xml=False, visible=False, width=1800):
    all_resized_imgs = []
    count=0
    classes_count = {}
    class_mapping = {}
    data_path = os.path.join(source_dir)
    jpg_names = [x for x in os.listdir(data_path) if x[-4:]=='.jpg']
    jpg_names.sort()
    xml_names = [x for x in os.listdir(data_path) if x[-4:]=='.xml']
    xml_names.sort()
    print(len(jpg_names))
    assert len(jpg_names) == len(xml_names)
    print(len(jpg_names))
    for i in range(len(jpg_names)):
        if(i<10840):
            continue
        assert jpg_names[i][:-4] == xml_names[i][:-4]
        print(str(i)+"  "+jpg_names[i][:-4])
        img_path = os.path.join(data_path, jpg_names[i])
        annot_path = os.path.join(data_path, xml_names[i])
        et = ET.parse(annot_path)
        element = et.getroot()
        element_objs = element.findall('object')
<<<<<<< HEAD:keras_frcnn/utils.py
         
        ##读取图片，并压缩图片
=======
                
         ##读取图片，并压缩图片
>>>>>>> 8c8f50e3f0157b68f9105770f94b75ff7a65f348:keras_frcnn/flaw_img_split.py
        raw_img = cv2.imread(img_path)
        
        element_width = raw_img.shape[1];
        element_height = raw_img.shape[0];
#        print(element_height)
#        print(element_width)
        ratio = float(width)/element_width
        height = int(element_height * ratio)
        
       
        resized_img = cv2.resize(raw_img, (width, height))
#        print(resized_img.shape[0])
#        print(resized_img.shape[1])
        if len(element_objs) > 0:
            resized_img_data = {'resized_img':resized_img,'width':width,
                               'height':height,'bboxes':[],
                               'filepath':img_path, 'filename':jpg_names[i][:-4]}
        
        for element_obj in element_objs:
            class_name = element_obj.find('name').text[:2]
            if class_name in white_list:
                continue
            assert len(class_name)==2
            
            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1
            if class_name not in class_mapping:
                class_mapping[class_name] = len(class_mapping)
            if class_name in white_list:
                continue
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
<<<<<<< HEAD:keras_frcnn/utils.py
=======
                count=count+1
              
>>>>>>> 8c8f50e3f0157b68f9105770f94b75ff7a65f348:keras_frcnn/flaw_img_split.py
                save_path_tmp = save_path+sep+class_name
                if not os.path.exists(save_path_tmp):
                    os.makedirs(save_path_tmp)
                Image.fromarray(resized_img).save(save_path_tmp+sep+'resized_'+jpg_names[i]) 
<<<<<<< HEAD:keras_frcnn/utils.py
=======
             
>>>>>>> 8c8f50e3f0157b68f9105770f94b75ff7a65f348:keras_frcnn/flaw_img_split.py
            ## 是否保存缩放后的xml文件
            if save_xml:
                xml_create(resized_img_data, save_path_tmp, file_name_prefix='resized_')              
        all_resized_imgs.append(resized_img_data)
<<<<<<< HEAD:keras_frcnn/utils.py
    return all_resized_imgs, class_mapping, classes_count
=======
    print(classes_count)
    print(class_mapping)
    print(count)
    return all_resized_imgs
>>>>>>> 8c8f50e3f0157b68f9105770f94b75ff7a65f348:keras_frcnn/flaw_img_split.py

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


def classify(data_path, target_dir,
             target_list = ['65','66','67','70', '75','77','79','80']):
    classes_count = {}
    for name in target_list:
        path = os.path.join(data_path, name)
        jpg_names = [x for x in os.listdir(path) if x[-4:]=='.jpg']
        jpg_names.sort()
        xml_names = [x for x in os.listdir(path) if x[-4:]=='.xml']
        xml_names.sort()
        assert len(jpg_names) == len(xml_names)
        print(len(jpg_names))
        for i in range(len(jpg_names)):
            assert jpg_names[i][:-4] == xml_names[i][:-4]
            img_path = os.path.join(path, jpg_names[i])
            annot_path = os.path.join(path, xml_names[i])
            et = ET.parse(annot_path)
            element = et.getroot()
            element_objs = element.findall('object')
            classes_name = set()
            if len(element_objs) > 0:
                #print(len(element_objs))
                for element_obj in element_objs:
                    class_name = element_obj.find('name').text[:2]
                    #print(type(class_name))
                    assert(len(class_name)==2)
                    classes_name.add(class_name)
            if len(classes_name) > 0:    
                #print(len(classes_name))
                for class_name in classes_name:
                    if class_name not in target_list:
                        continue
                    if class_name not in classes_count:
                        classes_count[class_name] = 1
                    else:
                        classes_count[class_name] += 1
                    target_path = os.path.join(target_dir + sep + class_name)
                    #print(target_path)
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
                    shutil.copy(img_path, target_path)
                    shutil.copy(annot_path, target_path)
    return classes_count

def img_detail_count(data_path, target_classes):
    heights = {'greater':0,'less':0,'equal':0}
    gt_100 = 0;
    width = 0
    total = 0;
    for class_name in target_classes:
        path = os.path.join(data_path, class_name)
        jpg_names = [x for x in os.listdir(path) if x[-4:]=='.jpg']
        print(len(jpg_names))
        for jpg_name in jpg_names:
            img_path = os.path.join(path, jpg_name)
            #print(img_path)
            img = cv2.imread(img_path)
            #type(img)
            width = img.shape[1]
            height = img.shape[0]
            total += 1
            if height>width:
                if height-width > 200:
                    gt_100+=1
                heights['greater'] += 1
            elif height==width:
                heights['equal'] += 1
            else:
                heights['less'] += 1
    return heights, gt_100

def bboxes_details(data_dir, target_classes, min_length=900):
    hw_ratio = {}
    wh_ratio = {}
    heights = {}
    widths = {}
    areas = {}
    bboxes_count = 0
    bboxes_of_class_count = {}
    for class_name in target_classes:
            bboxes_of_class_count[class_name] = 0
    for class_name in target_classes:
        data_path = os.path.join(data_dir, class_name)
        xml_names = [x for x in os.listdir(data_path) if x[-4:]=='.xml']
        print(len(xml_names))
        for xml_name in xml_names:
            annot_path = os.path.join(data_path, xml_name)
            et = ET.parse(annot_path)
            element = et.getroot()
            element_objs = element.findall('object')
            if len(element_objs)>0:
                for element_obj in element_objs:
                    class_name = element_obj.find('name').text[:2]
                    if class_name in target_classes:        
                        obj_bbox = element_obj.find('bndbox')
                        bboxes_count += 1
                        bboxes_of_class_count[class_name] += 1
                        x1 = int(round(float(obj_bbox.find('xmin').text)))
                        y1 = int(round(float(obj_bbox.find('ymin').text)))
                        x2 = int(round(float(obj_bbox.find('xmax').text)))
                        y2 = int(round(float(obj_bbox.find('ymax').text)))
                        
                        width = x2 - x1
                        w = int(width / 10)
                        if w not in widths:
                            widths[w] = 1;
                        else:
                            widths[w] += 1
                            
                        height = y2 - y1
                        h = int(height / 10)
                        if h not in heights:
                            heights[h] = 1
                        else:
                            heights[h] += 1
                            
                        ratio = int(height / (width+0.00001))
                        if ratio not in hw_ratio:
                            hw_ratio[ratio] = 1
                        else:
                            hw_ratio[ratio] += 1
                        
                        r = int(width/(height+0.00001))
                        if r not in wh_ratio:
                            wh_ratio[r] = 1
                        else:
                            wh_ratio[r] += 1
                            
                        area = width * height
                        a = int(area / 1000)
                        if a not in areas:
                            areas[a] = 1
                        else:
                            areas[a] += 1
    plt.figure(1) 
    height_x_label = heights.keys()
    height_y_label = heights.values()
    plt.plot(height_x_label, height_y_label)
    plt.xlabel("(*10)")
    plt.ylabel("count of each height")
    plt.title("the count of each heigth, total="+str(bboxes_count))
    plt.savefig("1200_full_height.jpg")
    
    plt.figure(2)
    width_x_label = widths.keys()
    width_y_label = widths.values()
    plt.plot(width_x_label, width_y_label)
    plt.xlabel("(*10)")
    plt.ylabel("count of each width")
    plt.title("the count of each width, total="+str(bboxes_count))
    plt.savefig("1200_width.jpg")
    return hw_ratio, wh_ratio, heights, widths,\
         areas, bboxes_count, bboxes_of_class_count
                        
            
        
if __name__=="__main__":
<<<<<<< HEAD:keras_frcnn/utils.py
    data_path = r'D:\dataset2018-05-23\raw_img_with_histogram'
    target_classes=['65','66','67','70', '75','77','79','80']
#    classify(data_path=data_path, 
#             target_dir=r'D:\dataset2018\raw_img_with_histogram_II')
    

    all_imgs = resize01(source_dir=r'D:\dataset2018-05-23\raw_img_with_histogram',
                        target_classes=target_classes, save=True, 
                        save_path=r'D:\dataset2018-05-23\resized_img_with_histogram\900_67_height', 
                        save_xml=True, visible=False, width=1200)
    all_imgs = augment(all_imgs, C)
    
#    heights, gt_100 = img_detail_count(
#            data_path=r'D:\dataset2018\resized_img_with_histogram\1200_67_height', 
#            target_classes=target_classes)
#    all_imgs, _, _ = resize02(data_path, save=True, 
#                       save_path=r'D:\dataset2018\resized', 
#                       save_xml=False, visible=True, width=900)
#    classes_count = classify(data_path, r'D:\dataset2018\raw_img_with_xml')
#    hw_ratio, wh_ratio, heights, widths, areas, bboxes_count, bboxes_of_class_count =\
#        bboxes_details(data_dir=data_path, target_classes=target_classes)
    
    
    
=======
    data_path = r'/opt/xdata/chaoy/dataset2018/'
    all_resized_imgs = resize02(data_path, save=True, 
                   save_path=r'/home/guopl/resized', 
                   save_xml=False, visible=False, width=900)
        
    
>>>>>>> 8c8f50e3f0157b68f9105770f94b75ff7a65f348:keras_frcnn/flaw_img_split.py
