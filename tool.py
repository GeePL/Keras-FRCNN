from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
from skimage import exposure
import time
import itertools
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os
from PIL import Image
from keras_frcnn import pascal_voc_parser
#os.environ['CUDA_VISIBLE_DEVICES']='0'
#os.system('echo $CUDA_VISIBLE_DEVICES')
import tensorflow as tf
import sys, time
import xml.etree.ElementTree as ET
 


class frcnn():
    def __init__(self, model_path, config_filename='./config.pickle', **kwargs):

        with open(config_filename, 'rb') as f_in:
            C = pickle.load(f_in)

        if C.network == 'resnet50':
            import keras_frcnn.resnet as nn
        elif C.network == 'vgg':
            import keras_frcnn.vgg as nn

        # turn off any data augmentation at test time
        C.use_horizontal_flips = False
        C.use_vertical_flips = False
        C.rot_90 = False
        C.model_path=model_path

        class_mapping = C.class_mapping

        if 'bg' not in class_mapping:
            class_mapping['bg'] = len(class_mapping)

        class_mapping = {v: k for k, v in class_mapping.items()}
        print(class_mapping)
        self.class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
        C.num_rois = 32
        C.class_mapping = class_mapping
        if C.network == 'resnet50':
            num_features = 1024
        elif C.network == 'vgg':
            num_features = 512

        if K.image_dim_ordering() == 'th':
            input_shape_img = (3, None, None)
            input_shape_features = (num_features, None, None)
        else:
            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, num_features)


        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)

        classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

        model_rpn = Model(img_input, rpn_layers)
        model_classifier_only = Model([feature_map_input, roi_input], classifier)

        model_classifier = Model([feature_map_input, roi_input], classifier)

        print('Loading weights from {}'.format(C.model_path))
        model_rpn.load_weights(C.model_path, by_name=True)
        model_classifier.load_weights(C.model_path, by_name=True)

        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')
        
        self.C = C
        self.model_rpn = model_rpn
        self.model_classifier_only = model_classifier_only

        


    def format_img_size(self, img, C):
        """ formats the image size based on config """
        img_min_side = float(C.im_size)
#        print img.shape
        (height,width,_) = img.shape

        if width <= height:
            ratio = img_min_side/width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side/height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio

    def format_img_channels(self, img, C):
        """ formats the image channels based on config """
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= C.img_channel_mean[0]
        img[:, :, 1] -= C.img_channel_mean[1]
        img[:, :, 2] -= C.img_channel_mean[2]
        img /= C.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

#控制短边在900
    def format_img(self, img, C):
        """ formats an image for model prediction based on config """
        img, ratio = self.format_img_size(img, C)
        img = self.format_img_channels(img, C)
        return img, ratio
#done
# Method to transform the coordinates of the bounding box to its original size
    def get_real_coordinates(self, ratio, x1, y1, x2, y2):
# //向下取整
        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))

        return (real_x1, real_y1, real_x2 ,real_y2)
    
    def add_location(self, img):
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.array([img,img,img])
        if img.shape[2] == 1:
            #img = np.concatenate((temp,temp,temp),axis=2)
            img = np.concatenate([img,img,img],axis=2)
        x_range = img.shape[0]
        y_range = img.shape[1]
        for x in range(x_range):
            for y in range(y_range):
                img[x,y,1]=int(255.0*x/x_range)
                img[x,y,2]=int(255.0*y/y_range)
        return img
    
    def remove_location(self, img):
        img = np.array(img)
        img[:,:,1] = img[:,:,0]
        img[:,:,2] = img[:,:,0]
        return img
    
# done    
    def getResultFromRealPre(self, real, pre, plot=True, ignoreClass=['66','68','69','80','81','67','75','74','76']):
        label= list(np.unique(['0']+pre+real))
        for igncls in ignoreClass:
            if igncls in label:
                label.remove(igncls)
        # cm  array, shape = [n_classes, n_classes]
        cm = confusion_matrix(real,pre,labels=label)
        result = self.getResultFromCm(cm, label)
        if plot:
            self.plot_confusion_matrix(cm, classes=label,
                              title=('acc_'+str(result['acc'])), save_img=True)
            plt.show()
        return result
# done        
    def getResultFromCm(self, cm, label):
        precision = {}
        recall = {}
        rate={}
        a=b=0
        for i in range(len(cm)):
            if sum(cm[:,i])!=0:
                a += cm[i,i]
                b += sum(cm[:,i])
                precision[label[i]]=1.0*cm[i,i]/sum(cm[:,i])
            if sum(cm[i,:])!=0:
                recall[label[i]]=1.0*cm[i,i]/sum(cm[i,:])
                # sum default axis=1
                # 每种瑕疵占总样本的比例
                rate[label[i]]=1.0*sum(cm[i,:])/sum(sum(cm))
        acc_all = 1.0*a/b
        TP=sum(sum(cm[1:,1:]))
        FN=sum(cm[1:,0])
        FP=sum(cm[0,1:])
        Rec = 1.0*TP/(TP+FN)
        Pre = 1.0*TP/(TP+FP)
        
        result={'rec':recall,'pre':precision,'rec_01':Rec,'pre_01':Pre,'acc':acc_all,'rate':rate}
        return result
    
    def detectWholeByXml(self, voc_path, save_path='./whole_voc/', save_all=False, save_failed=True, add_zero = True, img_set = 'test', bbox_threshold = 0.6, bmp_path = '/media/ccnt/hard disk/images_xmls/cy_whole/all/PNGs/'):
        all_imgs, classes_count, class_mapping=pascal_voc_parser.get_data(voc_path)
        val_imgs = [s for s in all_imgs if s['imageset'] == img_set]
        val_whole = []
        preflaws = []
        realflaws = []
        
        for val_img in val_imgs:
            filepath = val_img['filepath']
            filename = filepath.split('/')[-1]
            #filename = filename[:filename.rfind('_')]
            if not filename in val_whole:
                val_whole.append(filename)
        bmp_path = './cy_image/1900/BMPImages/'
        xml_path = './cy_image/1900/Annotations/'

        l = len(val_whole)
        i = 0
        t = time.time()
        for val_img in val_whole:
            i+= 1
            t0 = time.time()-t
            r_t = int(float(t0)/(i+1)*(l-i-1))
            sys.stdout.write('predicting val imgs: {}/{} using time: {}s, remain time: {}s    \r'.format(i + 1, l, int(t0), r_t))
            sys.stdout.flush()

            try:
                tree = ET.parse(xml_path+val_img+'.xml')
            except:
                print(xml_path+val_img+'.xml', ' read error')
                continue
            
            root = tree.getroot()
            element_objs = root.findall('object')
            realflaw=[]
            preflaw=[]
            for obj in element_objs:
                if obj is None:
                    continue
                elif obj.find('name').text is None:
                    continue
                realflaw.append(obj.find('name').text)
            realflaw = list(set(realflaw))
            img = cv2.imread(bmp_path+val_img+'.png')
            img, flaws = self.FastDetectWhole(img, bbox_threshold=bbox_threshold)
            preflaw = (list(set(flaws)))
            pre, real = self.flattenOneEasy(preflaw, realflaw)
            
            if (save_failed and (pre != real)):
                filepath = save_path+'/failed/'+str(real)+'/'
                filename = val_img+'?r'+str(real)+'?p'+str(pre)+'.jpg'
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                cv2.imwrite(filepath+filename, img)
            if save_all:
                filepath = save_path+'/all/'+str(real)+'/'
                filename = val_img+'?r'+str(real)+'?p'+str(pre)+'.jpg'
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                cv2.imwrite(filepath+filename, img)
            preflaws.append(pre)
            realflaws.append(real)
            
        if add_zero:
            zero_path = '/media/ccnt/hard disk/images_xmls/cy_whole/zeros/zero_test/'
            zeros = os.listdir(zero_path)
            
            whole_path = '/media/ccnt/hard disk/images_xmls/cy_whole/zeros/zero_raw/'
            tp_whole = np.array(os.listdir(whole_path))
            zero_whole = []
           
            for zero in zeros:
                tmh=zero[:zero.rfind('_')]
                if (not tmh in zero_whole) and (tmh+'.png' in tp_whole):
                    zero_whole.append(tmh)
            np.random.shuffle(zero_whole)
            i=1000
            zero_whole = zero_whole[:i]
            i = 0
            t = time.time()
            l = len(zero_whole)
            for zero in zero_whole:
                i+= 1
                t0 = time.time()-t
                r_t = int(float(t0)/(i+1)*(l-i-1))
                sys.stdout.write('predicting zero imgs: {}/{} using time: {}s, remain time: {}s     \r'.format(i + 1, l, int(t0), r_t))
                sys.stdout.flush()

                img = cv2.imread(whole_path+zero+'.png')
                img, flaws = self.FastDetectWhole(img, bbox_threshold=bbox_threshold)
                pre, real = self.flattenOneEasy(flaws, [])
                if (save_failed and (pre != real)):
                    filepath = save_path+'/failed/'+str(real)+'/'
                    filename = zero+'?r'+str(real)+'?p'+str(pre)+'.jpg'
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    cv2.imwrite(filepath+filename, img)
                if save_all:
                    filepath = save_path+'/all/'+str(real)+'/'
                    filename = zero+'?r'+str(real)+'?p'+str(pre)+'.jpg'
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    cv2.imwrite(filepath+filename, img)
                preflaws.append(pre)
                realflaws.append(real)

        return preflaws, realflaws, accuracy_score(realflaws,preflaws)
#load voc_xml using voc_path, detect imgs by img_set(eg. test), save false(predict failed) imgs or save all predicted imgs
#return 
    def detectSaveByXml(self, voc_path, add_zero = False, zero_path = '/home/ccnt/chaoyang/test_set/0/', save_path = '', img_set = 'test', save_failed = True, save_all = False, add_location = False, rate = 1.0, bbox_threshold = 0.6, max_boxes = 300):  
        print(voc_path)
        if save_path == '':
            save_path = voc_path + '/detect/'
        all_imgs, classes_count, class_mapping=pascal_voc_parser.get_data(voc_path)
        val_imgs = [s for s in all_imgs if s['imageset'] == img_set]
        preflaws = []
        realflaws = []
        l = len(val_imgs)
        i = 0
        t = time.time()
        for img in val_imgs:
            i+= 1
            t0 = time.time()-t
            r_t = int(float(t0)/(i+1)*(l-i-1))
            sys.stdout.write('predicting val imgs: {}/{} using time: {}s, remain time: {}s    \r'.format(i + 1, l, int(t0), r_t))
            sys.stdout.flush()
            fpath = img['filepath']
#            print(fpath)
#            print("**************")
            fname = fpath.split('\\')[-1]
#            print(fname)
#            print(fname[:-4])
            img_data = cv2.imread(fpath)
         
            img_data,flaws = self.detect(img_data, bbox_threshold = bbox_threshold, max_boxes = max_boxes)
            
            classes = []
            for bbox in img['bboxes']:
                classes.append(bbox['class'])
                textLabel = bbox['class']
                x1 = int(bbox['x1'])
                y1 = int(bbox['y1'])
                x2 = int(bbox['x2'])
                y2 = int(bbox['y2'])
                cv2.rectangle(img_data,(x1, y1), (x2, y2), (0,255,255),2)   
                textOrg = (x1, y1)
                cv2.putText(img_data, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
            classes = list(set(classes))

            pre, real = self.flattenOneEasy(flaws, classes)

            if (save_failed and (pre != real)):
                filepath = save_path+'/failed/'+str(real)+'/'
                filename = fname[:-4]+'real'+str(real)+'pre'+str(pre)+'.jpg'
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                Image.fromarray(img_data).save(filepath+filename)
            if save_all:
                filepath = save_path+'/all/'+str(real)+'/'
                filename = fname[:-4]+"real"+str(real)+"pre"+str(pre)+'.png'
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                Image.fromarray(img_data).save(filepath+filename)
          
            preflaws.append(pre)
            realflaws.append(real)
            
        if add_zero:
            zeros = os.listdir(zero_path)
            np.random.shuffle(zeros)
            zeros = zeros[:int(rate*len(val_imgs))]
            l = len(zeros)
            i = 0
            t = time.time()
            for zero in zeros:
                i+= 1
                t0 = time.time()-t
                r_t = int(float(t0)/(i+1)*(l-i-1))
                sys.stdout.write('predicting zero imgs: {}/{} using time: {}s, remain time: {}s    \r'.format(i + 1, l, int(t0), r_t))
                sys.stdout.flush()
                img_data = cv2.imread(zero_path+zero)
                img_data,flaws = self.detect(img_data, bbox_threshold = bbox_threshold)                
                pre, real = self.flattenOneEasy(flaws, [])
                
                if (save_failed and (pre != real)):
                    filepath = save_path+'/failed/'+str(real)+'/'
                    filename = zero[:-4]+'real'+str(real)+'pre'+str(pre)+'.png'
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    Image.fromarray(img_data).save(filepath+filename)
                if save_all:
                    filepath = save_path+'/all/'+str(real)+'/'
                    filename = zero[:-4]+'real'+str(real)+'pre'+str(pre)+'.png'
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    Image.fromarray(img_data).save(filepath+filename)
                    
                preflaws.append(pre)
                realflaws.append(real)
        return preflaws, realflaws, accuracy_score(realflaws,preflaws)
                    
    def detectSaveByFiles(self, path_read, path_save='', save_failed = True, save_all = True, add_location = False):
        if path_save == '':
            temp = path_read
            path_save = temp.replace(temp.split('/')[-2],temp.split('/')[-2]+'_result')
        preflaws = []
        realflaws = []
        for fpathe,dirs,fs in os.walk(path_read):
            for f in fs:
                read_name = os.path.join(fpathe,f)
                img_data = cv2.imread(read_name)
                img_data,flaws = self.detect(img_data, add_location = add_location)
                pre, real = self.flattenOneEasy(flaws, [fpathe.split('/')[-1]])

                if (save_failed and (pre != real)):
                    filepath = path_save+'/failed/'+str(real)+'/'
                    filename = f[:-4]+'?r'+str(real)+'?p'+str(pre)+'.jpg'
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    Image.fromarray(img_data).save(filepath+filename)
                if save_all:
                    filepath = path_save+'/all/'+str(real)+'/'
                    filename = f[:-4]+'?r'+str(real)+'?p'+str(pre)+'.jpg'
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    Image.fromarray(img_data).save(filepath+filename)
                
                preflaws.append(pre)
                realflaws.append(real)
        return preflaws, realflaws, accuracy_score(realflaws,preflaws)
            
#detect and save imgs
#input[ path_read: path of imgs to be read, path_save: path of imgs to be save after detect ]
#output[]                     
#    def detect_save(self, path_read, path_save, save = True):
#        for fpathe,dirs,fs in os.walk(path_read):
#            for f in fs:
#                read_name = os.path.join(fpathe,f)
#                save_path = fpathe.replace(path_read,path_save)
#                save_name = os.path.join(save_path,f)
#                if not os.path.exists(save_path):
#                    os.makedirs(save_path)
#                img = cv2.imread(read_name)
#                img,flaws = self.detect(img) 
#                flaw = ''
#                if len(flaws) == 0:
#                    flaw = '?0'
#                else:
#                    for i in set(flaws):
#                        flaw = flaw + '?' + str(i)
#                if save:
#                    Image.fromarray(img).save(save_name[:-4]+flaw+'.jpg')

                    
    def deleteclasses(self, lst, classes):
        for cls in classes:
            for i in lst:
                if cls in i:
                    i.remove(cls)
        return lst
    
    def detectWholeByFiles(self, img_path, xml_path, save_path, save_all=False, save_failed=True, zero_path = '', bbox_threshold=0.6):
        names = os.listdir(img_path)
        l = len(names)
        i = 0
        t = time.time()
        preflaws = []
        realflaws = []
        for name in names:
            img = cv2.imread(img_path+name)
            if img is None:
                continue
            i+= 1
            t0 = time.time()-t
            r_t = int(float(t0)/(i+1)*(l-i-1))
            sys.stdout.write('predicting val imgs: {}/{} using time: {}s, remain time: {}s    \r'.format(i + 1, l, int(t0), r_t))
            sys.stdout.flush()

            try:
                tree = ET.parse(xml_path+name[:-4]+'.xml')
            except:
                print(xml_path+name[:-4]+'.xml', ' read error')
                continue
            
            root = tree.getroot()
            element_objs = root.findall('object')
            realflaw=[]
            preflaw=[]
            for obj in element_objs:
                if obj is None:
                    continue
                elif obj.find('name').text is None:
                    continue
                realflaw.append(obj.find('name').text)
    
            realflaw = list(set(realflaw))            
            img, flaws = self.FastDetectWhole(img,bbox_threshold=bbox_threshold)
            preflaw = (list(set(flaws)))
            pre, real = self.flattenOneEasy(preflaw, realflaw)
            
           
            for element_obj in element_objs:
                obj_bbox = element_obj.find('bndbox')
                x1 = int(round(float(obj_bbox.find('xmin').text)))
                y1 = int(round(float(obj_bbox.find('ymin').text)))
                x2 = int(round(float(obj_bbox.find('xmax').text)))
                y2 = int(round(float(obj_bbox.find('ymax').text)))
                cls = element_obj.find('name').text
                cv2.rectangle(img,(x1, y1), (x2, y2), (0,0,255),2)           
                textLabel = cls          
                textOrg = (x1, y1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
            
            if (save_failed and (pre != real)):
                filepath = save_path+'/failed/'+str(real)+'/'
                filename = name[:-4]+'?r'+str(real)+'?p'+str(pre)+'.jpg'
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                cv2.imwrite(filepath+filename, img)
            if save_all:
                filepath = save_path+'/all/'+str(real)+'/'
                filename = name[:-4]+'?r'+str(real)+'?p'+str(pre)+'.jpg'
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                cv2.imwrite(filepath+filename, img)
            preflaws.append(pre)
            realflaws.append(real)
            
        if zero_path != '':
            zeros = os.listdir(zero_path)                    
            i = 0
            t = time.time()
            l = len(zeros)
            for zero in zeros:
                i+= 1
                t0 = time.time()-t
                r_t = int(float(t0)/(i+1)*(l-i-1))
                sys.stdout.write('predicting zero imgs: {}/{} using time: {}s, remain time: {}s     \r'.format(i + 1, l, int(t0), r_t))
                sys.stdout.flush()

                img = cv2.imread(zero_path+zero)
                img, flaws = self.FastDetectWhole(img,bbox_threshold=bbox_threshold)
                pre, real = self.flattenOneEasy(flaws, [])
                if (save_failed and (pre != real)):
                    filepath = save_path+'/failed/'+str(real)+'/'
                    filename = zero[:-4]+'?r'+str(real)+'?p'+str(pre)+'.jpg'
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    cv2.imwrite(filepath+filename, img)
                if save_all:
                    filepath = save_path+'/all/'+str(real)+'/'
                    filename = zero[:-4]+'?r'+str(real)+'?p'+str(pre)+'.jpg'
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    cv2.imwrite(filepath+filename, img)
                preflaws.append(pre)
                realflaws.append(real)

        return preflaws, realflaws, accuracy_score(realflaws,preflaws)
    
                
                
    def FastDetectWhole(self, img, h_m = 1900, bbox_threshold=0.6):
        img = np.array(img)
        return_img = img.copy()
        flaws = []
        h = img.shape[0]
        start = 0
        while start < h:
            if start + h_m > h:
                start = h - h_m
            m_img = img[start:start+h_m,:,:]            
            m_img, flaw = self.detect(m_img, bbox_threshold=bbox_threshold)
            if flaw != []:
                img[start:start+h_m,:,:] = m_img
                flaws = flaws + flaw
            start += h_m
        return img, flaws         

    def detectWhole(self, img, gamma = 1.0, add_location = False, rate = [1900,1200]):
        if add_location:
            img = self.add_location(img)
        img = np.array(img)
        img = Image.fromarray(img)
        height = img.height
        imgs, begin_heights=self.seperate(img,rate)
        if len(begin_heights) > 1:
            last_height = begin_heights[1] - (height % begin_heights[1])
        newimgs = []
        flaws = []
        for img in imgs:
            if gamma != 1.0:
                img = exposure.adjust_gamma(np.array(img),gamma)
            img,flaw = self.detect(img, add_location = add_location)
            newimgs.append(img)
            flaws = flaws + flaw
        img = []
        for idx in range(len(newimgs)):
            if img == []:
                img = newimgs[0]
            elif idx != len(newimgs)-1:
                img = np.concatenate((img,newimgs[idx]))
            else:
                img = np.concatenate((img,newimgs[idx][last_height:,:,:]))
        return img,flaws

    def seperate(self, img, rate=[1900,1900]):
        width = rate[0]
        height = rate[1]
        img_width = img.width
        img_height = img.height
        cut_height=int(1.0*img_width/width*height)
        i = 0
        imgs = []
        begin_heights = []
        while True:
            if (i+1) * cut_height <= img_height:
                imgs.append(img.crop((0,i*cut_height,img_width,(i+1)*cut_height)))
                begin_heights.append(i*cut_height)
            else:
                imgs.append(img.crop((0,max(img_height-cut_height,0),img_width,img_height)))
                begin_heights.append(max(img_height-cut_height,0))
                return imgs, begin_heights
            i = i+1
         
#example: input [ [[61,63]], [[61]] ], output [ [[61],[63]], [[61],[61]] ]    
    def flatten(self, lstP,lstR):
        newlstP=[]
        newlstR=[]
        for idx in range(len(lstP)):
            l1 = lstP[idx]
            l2 = lstR[idx]
            if len(l1) == 0:
                l1 = ['0']
            if len(l2) == 0:
                l2 = ['0']
            x1 = l1[0]
            x2 = l1[0]
            if set(l1) != set(l2):
                for l in l2:
                    if l != x1:
                        x2 = l
                        break                        
            newlstP.append(x1)
            newlstR.append(x2)
            
        return newlstP, newlstR
#example: input [ [[61,63]], [[61]] ], output [ [[61]], [[61]] ]  
    def flatten_easy(self, lstP,lstR):
        newlstP=[]
        newlstR=[]
        for idx in range(len(lstP)):
            l1 = lstP[idx]
            l2 = lstR[idx]
            if len(l1) == 0:
                l1 = ['0']
            if len(l2) == 0:
                l2 = ['0']
            x1 = l1[0]
            x2 = l2[0]
            for l in l1:
                if l in l2:
                    x1 = l
                    x2 = l
                    break
            newlstP.append(x1)
            newlstR.append(x2)                         
        return newlstP, newlstR
    
    def flattenOneEasy(self, pre, real):
        if len(pre) == 0:
            pre = ['0']
        if len(real) == 0:
            real = ['0']
        x1 = pre[0]
        x2 = real[0]
        for p in pre:
            if p in real:
                x1 = p
                x2 = p
                break
        return x1, x2
    
 #images departed in flaw names   
    def getImgFlaws(self, path):
        imgs=[]
        realflaws=[]
        flaws = os.listdir(path)
        for flaw in flaws:
            if flaw.isdigit():
                newpath = path+'/'+flaw+'/'
                names = os.listdir(newpath)
                for name in names:
                    imgs.append(cv2.imread(newpath+name))
                    realflaws.append([flaw])
        return imgs, realflaws
    
    def accUsePath(self, path, ignoreClasses=['610','670']):
        imgs, realflaws = self.getImgFlaws(path)
        return self.acc(imgs, realflaws, True, ignoreClasses)
     
    def TrainingWholeAcc(self, val_imgs, C, model_rpn, model_classifier, inv_map, bmp_path = '/media/ccnt/hard disk/images_xmls/cy_whole/all/BMPs/', easy = True, ignoreClasses=['610','670'], add_zero = True, zero_weight = 1):
        preflaws = []
        realflaws = []
        img_filenames = []
        realflaws_dict = {}
        preflaws_dict = {}
        h_m = 1900
        t = time.time()
        l = len(val_imgs)
        i = 0
        for val_img in val_imgs:
            if val_img['filepath'].split('/')[-2] == 'zero_val':
                continue
            filepath = val_img['filepath']
            filename = filepath.split('/')[-1]
            filename = filename[:filename.rfind('_')]+'.bmp'
            if not realflaws_dict.has_key(filename):
                realflaws_dict[filename] = []
            imgflaw=[]
            for bbox in val_img['bboxes']:
                imgflaw.append(bbox['class'])
            realflaws_dict[filename] += list(set(imgflaw))
            
            if not preflaws_dict.has_key(filename):
                preflaws_dict[filename] = []               
                img = cv2.imread(bmp_path+filename)
                flaws = []
                h = img.shape[0]
                start = 0
                while start < h:
                    if start + h_m > h:
                        start = h - h_m
                    m_img = img[start:start+h_m,:,:]            
                    flaw = self.getflaw(img, C, model_rpn, model_classifier, inv_map)
                    flaws += (list(set(flaw)))
                    start += h_m
                realflaws_dict[filename] = (list(set(flaws)))
            i+= 1
            t0 = time.time()-t
            r_t = int(float(t0)/(i+1)*(l-i-1))
            sys.stdout.write('predicting val imgs: {}/{} using time: {}s, remain time: {}s\r'.format(i + 1, l, int(t0), r_t))
            sys.stdout.flush()
        for i in realflaws_dict.keys():
            realflaws.append(list(set(realflaws_dict[i])))
            preflaws.append(list(set(preflaws_dict[i])))
        
        if add_zero:
            zero_path = '/media/ccnt/hard disk/images_xmls/cy_whole/zeros/zero_val/'
            zeros = os.listdir(zero_path)
            whole_path = '/media/ccnt/hard disk/images_xmls/cy_whole/zeros/zero_raw/'
            tp = []
            i = 0
            t = time.time()
            l = len(zeros)
            for zero in zeros:
                zero = zero[:-6] + '.bmp'
                if zero not in tp:
                    tp.append(zero)
                    img = cv2.imread(whole_path+zero)
                    flaws = []
                    h = img.shape[0]
                    start = 0
                    while start < h:
                        if start + h_m > h:
                            start = h - h_m
                        m_img = img[start:start+h_m,:,:]            
                        flaw = self.getflaw(img, C, model_rpn, model_classifier, inv_map)
                        flaws += (list(set(flaw)))
                        start += h_m
                    realflaws.append([])
                    preflaws.append(list(set(flaws)))
                i+= 1
                t0 = time.time()-t
                r_t = int(float(t0)/(i+1)*(l-i-1))
                sys.stdout.write('predicting zero imgs: {}/{} using time: {}s, remain time: {}s     \r'.format(i + 1, l, int(t0), r_t))
                sys.stdout.flush()
        
        realflaws = self.deleteclasses(realflaws, ignoreClasses)
        preflaws = self.deleteclasses(preflaws, ignoreClasses)
        if easy:
            preflaws, realflaws = self.flatten_easy(preflaws, realflaws) 
        else:
            preflaws, realflaws = self.flatten(preflaws, realflaws)
        sample_weight = np.where( np.array(realflaws)=='0', zero_weight, 1)
        cm = self.cm(realflaws, preflaws, ignoreClasses,save_img = False)
        label= list(np.unique(['0']+realflaws+preflaws))
        acc = {}
        a=0
        b=0
        for i in range(len(cm)):
            if sum(cm[i,:])!=0:
                a += cm[i,i]
                b += sum(cm[i,:])
                acc[label[i]]=1.0*cm[i,i]/sum(cm[i,:])
        acc['all'] = 1.0*a/b
        return acc, accuracy_score(realflaws,preflaws,sample_weight = sample_weight)

        
#same with acc, but the detect function is different    
    def accWhileTrain(self, val_imgs, C, model_rpn, model_classifier, inv_map, easy = True, ignoreClasses=['610','670'], zero_weight = 1, rate = 1.0, bbox_threshold = 0.5):
        preflaws = []
        realflaws = []
        zeros=[]
        flaws=[]
        for i in val_imgs:
            if i['bboxes'] == []:
                zeros.append(i)
            else:
                flaws.append(i)
        np.random.shuffle(zeros)
        zeros = zeros[:int(rate*len(flaws))]
        new_val_imgs = flaws + zeros
        
        l = len(new_val_imgs)
        i = 0
        t = time.time()
        for val_img in new_val_imgs:
            i+= 1
            t0 = time.time()-t
            r_t = int(float(t0)/(i+1)*(l-i-1))
            sys.stdout.write('predicting val imgs: {}/{} using time: {}s, remain time: {}s    \r'.format(i + 1, l, int(t0), r_t))
            sys.stdout.flush()
            imgflaw=[]
            for bbox in val_img['bboxes']:
                imgflaw.append(bbox['class'])
            realflaws.append(list(np.unique(imgflaw)))
            
            img = cv2.imread(val_img['filepath'])
            flaws = self.getflaw(img, C, model_rpn, model_classifier, inv_map, bbox_threshold = bbox_threshold )
            preflaws.append(list(np.unique(flaws)))                    

        realflaws = self.deleteclasses(realflaws, ignoreClasses)
        preflaws = self.deleteclasses(preflaws, ignoreClasses)
        if easy:
            preflaws, realflaws = self.flatten_easy(preflaws, realflaws) 
        else:
            preflaws, realflaws = self.flatten(preflaws, realflaws)
        sample_weight = np.where( np.array(realflaws)=='0', zero_weight, 1)       
        tp=self.getResultFromRealPre( pre=preflaws, real=realflaws, ignoreClass =['66','68','69','80','70','78','79','65','74','76'] )
        acc = (tp['rec']['0'] + 0.4*tp['rec']['63'] + tp['rec']['73'] + 0.2*tp['rec']['77'])/3
        acc1 = (tp['rec_01'] + 3*tp['rec']['0'] + tp['rec']['61'] + tp['rec']['72'] + tp['rec']['73'])/7

        return tp['rec'], acc, acc1
        
# done    
    def acc(self, imgs, realflaws, easy = True, ignoreClasses=['610','670']):   
        preflaws=[]
        for img in imgs:
            # return: img:经过画框的图,flaws
            img,flaws = self.detect(img) 
            preflaws.append(list(np.unique(flaws)))
        realflaws = self.deleteclasses(realflaws, ignoreClasses)
        preflaws = self.deleteclasses(preflaws, ignoreClasses)
        if easy:
            preflaws, realflaws = self.flatten_easy(preflaws, realflaws) 
        else:
            preflaws, realflaws = self.flatten(preflaws, realflaws) 
        return realflaws, preflaws, accuracy_score(realflaws,preflaws)

#confusion matrix
# done
    def cm(self, realflaws, preflaws, ignoreClasses=['610','670'], save_img = True):
        label= list(np.unique(['0']+preflaws+realflaws))
        for igncls in ignoreClasses:
            if igncls in label:
                label.remove(igncls)
        # acc 返回正确分类的比例
        acc = accuracy_score(realflaws,preflaws)  
        cm = confusion_matrix(realflaws,preflaws,labels=label)
        self.plot_confusion_matrix(cm, classes=label,
                              title=('acc_'+str(acc)), save_img=save_img)
        plt.show()
        return cm

# done   
    def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          save_img = True,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
    """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        if save_img:
            plt.savefig(title+'.png')

# done
    def getflaw(self, img, C, model_rpn, model_classifier, inv_map, bbox_threshold = 0.5 ):
        if img is None:
            return []
        X, ratio = self.format_img(img, C)
        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))
        [Y1, Y2] = model_rpn.predict(X)
        #why takes so much time when rpn_to_roi?
        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        bboxes = {}
        probs = {}
        
        for jk in range(R.shape[0]//C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//C.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier.predict([X, ROIs])
            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = inv_map[np.argmax(P_cls[0, ii, :])]
                
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]

                    cls_num = np.argmax(P_cls[0, ii, :])
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                        tx /= C.classifier_regr_std[0]
                        ty /= C.classifier_regr_std[1]
                        tw /= C.classifier_regr_std[2]
                        th /= C.classifier_regr_std[3]
                        x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass

                    bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))
                    
        return bboxes.keys()


# done
# return: img:经过画框的图,flaws
    def detect(self,img, add_location = False, return_img_type = 'img', bbox_threshold = 0.5, max_boxes = 300, overlap_thresh = 0.7):
        img = np.array(img)
        if img is None:
            return [],[]
        if add_location:
            img = self.add_location(img)
        C = self.C
        model_rpn = self.model_rpn
        model_classifier_only = self.model_classifier_only

        all_imgs = []
        location = []

        classes = {}
        t1 = time.time()

        visualise = True

        st = time.time()

        X, ratio = self.format_img(img, C)
        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)
#        print 't1: ', time.time()-t1
     
        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=overlap_thresh, max_boxes=max_boxes)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        # resized后的框
        bboxes = {}
        # resized前的框
        return_bboxes = {}
        probs = {}
 #       print 't2: ', time.time()-t1
        for jk in range(R.shape[0]//C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//C.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
#            print 't3: ', time.time()-t1
            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = C.class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    return_bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                return_bboxes[cls_name].append(self.get_real_coordinates(ratio, C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)))
                probs[cls_name].append(np.max(P_cls[0, ii, :]))
#        return bboxes.keys()
        pos_img = np.zeros((1900,1900))
        flaws=[]
        class_to_color = self.class_to_color
#        print 't4: ', time.time()-t1
        for key in bboxes:

            bbox = np.array(bboxes[key])
            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            return_bboxes[key], probs[key] = roi_helpers.non_max_suppression_fast(np.array(return_bboxes[key]), np.array(probs[key]), overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]
                # 可以尝试修改为return_bboxes
                (real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)
                cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
                # ?????????????????????????????????
                pos_img[real_x1:real_x2, real_y1:real_y2] += new_probs[jk]
                textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))         

                flaws.append(str(key))
                (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                textOrg = (real_x1, real_y1-0)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

#        print('Elapsed time = {}'.format(time.time() - st))

#        print 't5: ', time.time()-t1
        if add_location:
            img = self.remove_location(img)
        if return_img_type == 'img':
            return img, flaws
        elif return_img_type == 'box':
            return return_bboxes, probs
#never used        
#compute pred,gt boxes, for prediction result analysis        
    def compute_pred_gt(self, voc_path, img_set = 'test', bbox_threshold = 0.5, RPN = False, max_boxes = 3):
        all_imgs, classes_count, class_mapping=pascal_voc_parser.get_data(voc_path)
        val_imgs = [s for s in all_imgs if s['imageset'] == img_set]
        pred = []
        gt = []
        l = len(val_imgs)
        i = 0
        t = time.time()
        #遍历测试集
        for img in val_imgs:
            #一张图片中所有的预测框
            pred_img = []
            #一张图片中的所有gt框
            gt_img = []
         
            fpath = img['filepath']
            fname = fpath.split('/')[-1]

            for bbox in img['bboxes']:  
                gt_one = {}
                gt_one['class'] = bbox['class']
                gt_one['bbox'] = [int(bbox['x1']),int(bbox['y1']),int(bbox['x2']),int(bbox['y2'])]
                gt_one['w'] = int(bbox['x2']) - int(bbox['x1'])
                gt_one['h'] = int(bbox['y2']) - int(bbox['y1'])
                gt_one['area'] = gt_one['w'] * gt_one['h']
                gt_one['max_iou'] = -1.0
                gt_one['name'] = fname
                gt_img.append(gt_one)
                
            img_data = cv2.imread(fpath)
            if not RPN:
                #模型返回的最终结果，框{name,boxes,}，概率
                return_bboxes, probs = self.detect(img_data, return_img_type = 'box', bbox_threshold = bbox_threshold)
            else:
                R, prob = self.get_RPN_output(img_data, max_boxes, overlap_thresh=0.7, p=True)
                return_bboxes = {}
                probs = {}
                return_bboxes['RPN'] = R
                probs['RPN'] = prob
                
            for key in return_bboxes:
                boxes = return_bboxes[key]
                prob = probs[key]
                for idx in range(len(boxes)):
                    pred_one = {}
                    [x1,y1,x2,y2] = boxes[idx]
                    pred_one['class'] = key
                    pred_one['bbox'] = boxes[idx]
                    pred_one['w'] = x2-x1
                    pred_one['h'] = y2-y1
                    pred_one['area'] = pred_one['w'] * pred_one['h']
                    pred_one['iou'] = -1.0
                    pred_one['iou_cls'] = -1.0
                    pred_one['prob'] = prob[idx]
                    pred_one['name'] = fname
                    pred_img.append(pred_one)
            pred_img, gt_img = self.deal_pred_gt(pred_img, gt_img) 
            i+= 1
            t0 = time.time()-t
            r_t = int(float(t0)/(i+1)*(l-i-1))
            sys.stdout.write('predicting val imgs: {}/{} using time: {}s, remain time: {}s    \r'.format(i + 1, l, int(t0), r_t))
            sys.stdout.flush()
            pred = pred + pred_img
            gt = gt + gt_img
        return pred, gt

