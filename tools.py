import json
import os
import shutil
import sys

import cv2
import scipy.io
import numpy as np
from PIL import Image
import pandas as pd

def replace_suffix(filename, suffix):
    portion = os.path.splitext(filename)
    if portion[1] != suffix:
        # print(portion[1])
        newname = portion[0] + suffix
    return newname
"按照缩放的边长对图片等比例缩放，并转成正方形居中"
def scale_img(img,scale_side,img_mode="RGB"):
    # "获得图片宽高"
    w1, h1 = img.size
    # print(w1,h1)
    # "根据最大边长缩放,图像只会被缩小，不会变大"
    # "当被缩放的图片宽和高都小于缩放尺寸的时候，图像不变"
    img.thumbnail((scale_side, scale_side))
    # "获得缩放后的宽高"
    w2, h2 = img.size
    # print(w2,h2)
    # "获得缩放后的比例"
    # s1 = w1 / w2
    # s2 = h1 / h2
    # s = (s1 + s2) / 2
    # "新建一张scale_side*scale_side的空白黑色背景图片"
    bg_img = Image.new(img_mode, (scale_side, scale_side), 0)
    # "根据缩放后的宽高粘贴图像到背景图上"
    if w2 == scale_side:
        bg_img.paste(img, (0, int((scale_side - h2) / 2)))
    elif h2 == scale_side:
        bg_img.paste(img, (int((scale_side - w2) / 2), 0))
    # "原图比缩放后的图要小的时候"
    else:
        bg_img.paste(img, (int((scale_side - w2) / 2), (int((scale_side - h2) / 2))))
    return bg_img

def save_array_as_csv(array_data,save_full_path):
    data1 = pd.DataFrame(array_data)
    data1.to_csv(save_full_path,header=False,index=False)

#一定注意是先高后宽img_size_h_w
def get_point_size_img(image,img_size_h_w):
    h0, w0 = image.shape[:2]  # orig hw
    # img_size_max=max(img_size_h_w[0],img_size_h_w[1])
    r1=img_size_h_w[0]/h0
    r2=img_size_h_w[1]/w0
    r=r2 if(r1>r2) else r1
    # r = img_size_max / max(h0, w0)  # resize image to img_size
    img = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
    # print(img.shape)
    img, (offset_y,offset_x)=fill_img(img,img_size_h_w)
    return img, r, (offset_y,offset_x)  # img, hw_original, hw_resized

def fill_img(img,img_size_h_w):
    now_img_h,now_img_w,_=img.shape
    assert(now_img_h<=img_size_h_w[0] and now_img_w<=img_size_h_w[1])
    empty_img=np.zeros((img_size_h_w[0],img_size_h_w[1],3),dtype=img.dtype)
    offset_y=(img_size_h_w[0]-now_img_h)//2
    offset_x=(img_size_h_w[1]-now_img_w)//2
    empty_img[offset_y:offset_y+now_img_h,offset_x:offset_x+now_img_w,:]=img
    return empty_img,(offset_y,offset_x)
    pass
###yolo5的letterbox
def letterbox(img, new_shape=(640, 640), color=(0, 0, 0), auto=True, scaleFill=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
def check_is_inside(point,shape):
    x,y=point
    h,w,_=shape
    is_in=True
    if(x<0 or x>=w or y<0 or y>=h):
        is_in=False
    return is_in
def get_json_data(json_path):
    data_array=[]
    with open(json_path) as f:
        json_dict = json.loads(f.read())
        # print(json_dict)
        w, h = json_dict["imageWidth"], json_dict["imageHeight"]
        for data in json_dict["shapes"]:
            # print(data)
            points = data["points"]
            data_array.append(points[0])
    return data_array

def get_mat_data(mat_path):
    """ 提取标注数据(相对坐标)  """
    data = scipy.io.loadmat(mat_path)  # 读取mat文件
    # print(data.keys())
    return data["image_info"]

def cv_img_rgb(path):
    img = Image.open(path)#支持中文路径
    img = img.convert('RGB')
    #因为opencv读取是按照BGR的顺序，所以这里转换一下即可
    img_rgb=cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)#COLOR_BGR2RGB
    return img_rgb

def img_save(img, save_img_full_path):
    cv2.imwrite(save_img_full_path,img)
    # image = Image.fromarray(np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    # image.save(save_img_full_path)

def file_copy(src_path,dest_path):
    try:
        shutil.copyfile(src_path,dest_path)
    except IOError as e:
        print("Unable to copy file. %s" % e)
        exit(1)
    except:
        print("Unexpected error:", sys.exc_info())
        exit(1)
def file_del(file_path):
    try:
        os.remove(file_path)
    except:
        print("Unexpected error:", sys.exc_info())
        exit(1)

