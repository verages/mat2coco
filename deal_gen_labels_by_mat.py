# coding=utf-8
"""
功能
    - 根据原始标注数据生成生成训练集、验证集和测试集
"""
import math
import os
import random
import shutil

import cv2
import numpy
import numpy as np
import pandas as pd
import scipy.io as sio
import json
import base64

from PIL import Image

from deal_gen_labels_by_json import img_size_h_w, gauss_ksize, gauss_sigma, gen_json_points
from tools import get_mat_data, cv_img_rgb, get_point_size_img, save_array_as_csv, replace_suffix, img_save

point_size = 1
point_color = (0, 0, 255)  # BGR
thickness = 4  # 0 、4、8



if __name__ == '__main__':

    start_index=0#这个字段是中间断了继续才需要的
    #1280*720
    save_img_idx=0
    save_imgs_path="/Users/linzhihui/Documents/dataset/val"
    save_labels_path="/Users/linzhihui/Documents/dataset/label"
    source_imgs_arr=["/Users/linzhihui/Documents/dataset/train"]
    source_mats_arr=["/Users/linzhihui/Documents/dataset/train_mat"]
    for k,img_root_path in enumerate(source_imgs_arr):
        for file in os.listdir(img_root_path):
            img_path = os.path.join(img_root_path, file)
            mat_path=os.path.join(source_mats_arr[k], replace_suffix(file,".mat"))
            if(os.path.exists(mat_path)):
                save_img_idx += 1
                print(save_img_idx)
                if(save_img_idx<=start_index):continue
                # if (save_img_idx != 1943): continue
                mat_data=get_mat_data(mat_path)
                points_arr=mat_data[0][0][0][0][0]
                # exit()

                rows,columns=points_arr.shape
                img_array=cv_img_rgb(img_path)
                img_array_new,ratio, (offset_y,offset_x)=get_point_size_img(img_array,img_size_h_w)
                # print( img0_h,img0_w)
                # img_now_h, img_now_w,_=img_array_new.shape
                # print(img_now_h, img_now_w)
                # print(offset_y, offset_x)
                # exit()
                mat_img_array=np.zeros(img_array_new.shape,dtype=np.float32)
                h_new,w_new,_=img_array_new.shape
                points=[]
                for i in range(rows):
                    # cv2.circle(img_array, (int(points_arr[i][0]), int(points_arr[i][1])), point_size, point_color, thickness)
                    old_zoord_x_y=[int(points_arr[i][0]),int(points_arr[i][1])]
                    #换算，原坐标换算到新坐标
                    #1先换算缩放
                    new_old_zoord_x_y=[int(old_zoord_x_y[0]*ratio),int(old_zoord_x_y[1]*ratio)]
                    #2在换算偏移
                    new_old_zoord_x_y=[new_old_zoord_x_y[0]+offset_x,new_old_zoord_x_y[1]+offset_y]
                    if(new_old_zoord_x_y[1]>=h_new):new_old_zoord_x_y[1]=h_new-1
                    if(new_old_zoord_x_y[0]>=w_new):new_old_zoord_x_y[0]=w_new-1
                    mat_img_array[new_old_zoord_x_y[1],new_old_zoord_x_y[0],:]=1
                    points.append(new_old_zoord_x_y)
                # print(rows)
                # print(np.sum(mat_img_array))
                mat_img_array_new = cv2.GaussianBlur(mat_img_array, gauss_ksize, gauss_sigma)
                # cv2.imshow("1",img_array_new)
                # mat_img_array_new=mat_img_array_new*255
                # cv2.imshow("2", mat_img_array_new)
                # cv2.waitKey(0)
                # print(mat_img_array_new)

                save_img_full_path=os.path.join(save_imgs_path,str(file.split(".")[0])+".jpg")
                img_save(img_array_new,save_img_full_path)
                #save_csv_full_path=os.path.join(save_labels_path,str(save_img_idx)+".csv")
                #save_array_as_csv(mat_img_array_new[:,:,0],save_csv_full_path)
                gen_json_points(points,file.split(".")[0])
                # exit()
                # pass
