import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from keras_frcnn import config
def get_data(input_path):
	C = config.Config()
#	min_pixes = C.min_pixes
	all_imgs = []

	classes_count = {}

	class_mapping = {}

	visualise = False

	data_paths = [os.path.join(input_path,s) for s in ['VOC2007']]
	

	print('Parsing annotation files')

	for data_path in data_paths:

		annot_path = os.path.join(data_path, 'Annotations')
		imgs_path = os.path.join(data_path, 'JPEGImages')
		imgsets_path_trainval = os.path.join(data_path, 'ImageSets','Main','trainval.txt')
		imgsets_path_test = os.path.join(data_path, 'ImageSets','Main','test.txt')

		trainval_files = []
		test_files = []
		try:
			with open(imgsets_path_trainval) as f:
				for line in f:
					trainval_files.append(line.strip() + '.jpg')
		except Exception as e:
			print(e)

		try:
			with open(imgsets_path_test) as f:
				for line in f:
					test_files.append(line.strip() + '.jpg')
		except Exception as e:
			if data_path[-7:] == 'VOC2012':
				# this is expected, most pascal voc distibutions dont have the test.txt file
				pass
			else:
				print(e)
		
		annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
		idx = 0
#		sum_w_less_64 = 0
#		total_w=0
#		sum_xml = 0
#		sum_h_less_64=0
		for annot in annots:
#			sum_xml += 1
			try:
				idx += 1

				et = ET.parse(annot)
				element = et.getroot()

				element_objs = element.findall('object')
				element_filename = element.find('filename').text
				element_width = int(element.find('size').find('width').text)
				element_height = int(element.find('size').find('height').text)
				#print(element_filename)
				if len(element_objs) > 0:
					annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
									   'height': element_height, 'bboxes': []}

					if element_filename in trainval_files:
						annotation_data['imageset'] = 'trainval'
					elif element_filename in test_files:
						annotation_data['imageset'] = 'test'
					else:
						annotation_data['imageset'] = 'trainval'
				for element_obj in element_objs:
					class_name = element_obj.find('name').text
					if class_name not in classes_count:
						classes_count[class_name] = 1
					else:
						classes_count[class_name] += 1

					if class_name not in class_mapping:
						class_mapping[class_name] = len(class_mapping)

					obj_bbox = element_obj.find('bndbox')
					x1 = int(round(float(obj_bbox.find('xmin').text)))
					y1 = int(round(float(obj_bbox.find('ymin').text)))
					x2 = int(round(float(obj_bbox.find('xmax').text)))
					y2 = int(round(float(obj_bbox.find('ymax').text)))
					'''
					w = abs(x2-x1)
					h = abs(y2-y1)
#					total_w+=1
					if(h<min_pixes & w < min_pixes):
#						sum_w_less_64+=1
#						sum_h_less_64+=1
						x1 = np.maximum(1, int(0.5*(x1+x2)-0.5*min_pixes)) if x2<(element_width-min_pixes/3) else np.maximum(1,int(0.5*(x1+x2)-min_pixes))
						x2 = np.minimum(element_width, int(0.5*(x1+x2)+0.5*min_pixes)) if x1>min_pixes/3 else np.minimum(element_width,int(0.5*(x1+x2)+min_pixes))
						y1 = np.maximum(1, int(0.5*(y1+y2)-0.5*min_pixes)) if y2<(element_height-min_pixes/3) else np.maximum(1,int(0.5*(y1+y2)-min_pixes))
						y2 = np.minimum(element_height, int(0.5*(y1+y2)+0.5*min_pixes)) if y1>min_pixes/3 else np.minimum(element_height,int(0.5*(y1+y2)+min_pixes))
						w=abs(x2-x1)
						h=abs(y2-y1)
					elif(h<min_pixes & w>=min_pixes):
#						sum_h_less_64+=1
						y1 = np.maximum(1,int(y1-0.5*(min_pixes-h))) if y2<(element_height-min_pixes/3) else np.maximum(1,int(y1-(min_pixes-h)))
						y2 = np.minimum(element_height,int(y2+0.5*(min_pixes-h))) if y1>min_pixes/3 else np.minimum(element_height,int(y2+(min_pixes-h)))
						ratio = min_pixes/h
						x1 = np.maximum(1,int(x1-ratio*0.5*w)) if x2<(element_width-min_pixes/3) else np.maximum(1,int(x1-ratio*w))
						x2 = np.minimum(element_width, int(x2+0.5*ratio*w)) if x1>min_pixes/3 else np.minimum(element_width,int(x2+ratio*w))
						h = abs(y2-y1)
						w = abs(x2-x1)
					elif(w<min_pixes & h>=min_pixes):
#						sum_w_less_64+=1
						x1 = np.maximum(1,int(x1-0.5*(min_pixes-w))) if x2<(element_width-min_pixes/3) else np.maximum(1,int(x1-(min_pixes-w)))
						x2 = np.minimum(element_width, int(x2+0.5*(min_pixes-w))) if x1>min_pixes/3 else np.minimum(element_width, int(x2+(min_pixes-w)))
						ratio = min_pixes/w
						y1 = np.maximum(1,int(y1-ratio*0.5*h)) if y2<(element_width-min_pixes/3) else np.maximum(1,int(y1-ratio*h))
						y2 = np.minimum(element_height, int(y2+0.5*ratio*h)) if y1>min_pixes/3 else np.minimum(element_height,int(y2+ratio*h))
						w = abs(x2-x1)
						h = abs(y2-y1)
					else:
						pass
					'''
      
					difficulty = int(element_obj.find('difficult').text) == 1
					annotation_data['bboxes'].append(
						{'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
				all_imgs.append(annotation_data)
				

				if visualise:
					img = cv2.imread(annotation_data['filepath'])
					for bbox in annotation_data['bboxes']:
						cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
									  'x2'], bbox['y2']), (0, 0, 255))
					cv2.imshow('img', img)
					cv2.waitKey(0)

			except Exception as e:
				print(e)
				continue
#	print(sum_xml, total_w, sum_w_less_64, sum_h_less_64)
	return all_imgs, classes_count, class_mapping

if __name__=="__main__":
    all_imgs,_,_ = get_data(
            "C:\\Users\\GeePL\\Desktop\\VOCtrainval_06-Nov-2007\\VOCdevkit")
