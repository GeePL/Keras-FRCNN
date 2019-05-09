# -*- coding: utf-8 -*-
import xml.dom.minidom
import os

def xml_create(img_details, annot_save_path, file_name_prefix=''):  
    """
    img_details : {height, width, bboxes, filename}
    annot_save_path : 
    """
    if not os.path.exists(annot_save_path):
        os.makedirs(annot_save_path)
        
    height = img_details['height']
    width = img_details['width']
    depth = 1
    filename = img_details['filename']
#    folder = img_details['folder']
    bboxes = img_details['bboxes']
    
    doc = xml.dom.minidom.Document()
    
    root = doc.createElement('annotation')
    root.setAttribute('verified','no')
    doc.appendChild(root)
    
#    nodeFolder = doc.createElement('folder')
#    nodeFolder.appendChild(doc.createTextNode(folder))
    
    nodeFilename = doc.createElement('filename')
    nodeFilename.appendChild(doc.createTextNode(filename))
    
    nodeSize = doc.createElement('size')
    nodeWidth = doc.createElement('width')
    nodeWidth.appendChild(doc.createTextNode(str(width)))
    nodeHeight = doc.createElement('height')
    nodeHeight.appendChild(doc.createTextNode(str(height)))
    nodeDepth = doc.createElement('depth')
    nodeDepth.appendChild(doc.createTextNode(str(depth)))
    nodeSize.appendChild(nodeWidth)
    nodeSize.appendChild(nodeHeight)
    nodeSize.appendChild(nodeDepth)
    
#    root.appendChild(nodeFolder)
    root.appendChild(nodeFilename)
    root.appendChild(nodeSize)
    if bboxes:
        for bbox in bboxes:
            nodeObject = doc.createElement('object')
            nodeName = doc.createElement('name')
            nodeName.appendChild(doc.createTextNode(bbox['class']))
            nodePose = doc.createElement('pose')
            nodePose.appendChild(doc.createTextNode('Unspecified'))
            nodeTruncated = doc.createElement('truncated')
            nodeTruncated.appendChild(doc.createTextNode('0'))
            nodeDifficult = doc.createElement('Difficult')
            nodeDifficult.appendChild(doc.createTextNode('0'))
            nodeObject.appendChild(nodeName)
            nodeObject.appendChild(nodePose)
            nodeObject.appendChild(nodeTruncated)
            nodeObject.appendChild(nodeDifficult)
            nodeBndbox = doc.createElement('bndbox')
            nodeXmin = doc.createElement('xmin')
            nodeXmin.appendChild(doc.createTextNode(str(bbox['x1']))) 
            nodeYmin = doc.createElement('ymin')
            nodeYmin.appendChild(doc.createTextNode(str(bbox['y1'])))
            nodeXmax = doc.createElement('xmax')
            nodeXmax.appendChild(doc.createTextNode(str(bbox['x2'])))
            nodeYmax = doc.createElement('ymax')
            nodeYmax.appendChild(doc.createTextNode(str(bbox['y2'])))
            nodeBndbox.appendChild(nodeXmin)
            nodeBndbox.appendChild(nodeYmin)
            nodeBndbox.appendChild(nodeXmax)
            nodeBndbox.appendChild(nodeYmax)
            nodeObject.appendChild(nodeBndbox)
            root.appendChild(nodeObject)
    else:
        pass
    sep = os.sep
    fp = open(annot_save_path+sep+file_name_prefix+filename+'.xml', 'w')
    doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")

