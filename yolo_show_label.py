import numpy as np
import cv2
import matplotlib.pyplot as plt
 
from PIL import Image
 
 
'''人工标注框的还原'''
 
# # 加载coco类别名称
# def load_classes(path):
#     fp = open(path, "r")
#     names = fp.read().split("\n")[:-1]
#     return names
 
# classes = load_classes('data/coco.names')
classes = "person"
# 数据类型:[cls_index, cx, cy, w, h]
boxes = np.loadtxt('/Users/linzhihui/Documents/dataset/label/td134.txt').reshape(-1,5)
 
img_path = '/Users/linzhihui/Documents/dataset/yolo_img/td134.jpg'
image = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
 
# 人工标注框的绘制 [cx,cy,w,h]
h, w, _ = image.shape
boxes[:,[1,3]] = boxes[:,[1,3]]*w
boxes[:,[2,4]] = boxes[:,[2,4]]*h
 
for box in boxes:
    p1 = box[1:3] - box[3:5]/2
    p1 = p1.astype(int)
    p2 = box[1:3] + box[3:5]/2
    p2 = p2.astype(int)
    print(p1,p2) # box
    print(classes[int(box[0])]) # class_name
    cv2.rectangle(image, tuple(p1), tuple(p2),(255,0,0),1)
    # arg:图像，标签，坐标，标签字体，字体大小，字体颜色，字体厚度
    cv2.putText(image,classes[int(box[0])],tuple(p1),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,0,0),0)
 
    
plt.figure(figsize=(25,15))
plt.imshow(image)
plt.show()