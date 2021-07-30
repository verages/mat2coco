import json
import math
import os
import random
import sys

import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import pandas as pd
from imgaug import parameters as iap
from deal_gen_labels_by_json import gauss_ksize, gauss_sigma
from tools import replace_suffix, cv_img_rgb, file_copy, img_save, save_array_as_csv, check_is_inside, file_del


def change_coarse(imgs):
    random_size = random.randint(0, 1) / 5.0 + 0.4  # 0.4--0.6
    images_aug = iaa.CoarseDropout(p=0.1, size_percent=random_size)(images=imgs)
    return images_aug
#Coarse+水平翻转和垂直翻转
def change_1(imgs,is_coarse=True):
    if(is_coarse):imgs = change_coarse(imgs)
    is_lr=random.randint(0, 1)<0.5
    is_ud=random.randint(0, 1)<0.5
    if(is_lr):
        imgs = iaa.Fliplr(1.0)(images=imgs)
    if (is_ud):
        imgs = iaa.Flipud(1.0)(images=imgs)
    return imgs,(is_lr,is_ud)
# 2Coarse+旋转
def change_2(imgs,is_coarse=True):
    if (is_coarse): imgs = change_coarse(imgs)
    random_angle=random.randint(-45,45)
    imgs = iaa.Rotate(random_angle)(images=imgs)
    # new_zoord=get_new_rotate_zoord(imgs[0], (0, 0), random_angle)
    # print(new_zoord)
    return imgs,random_angle
# 3Coarse+彩色噪声(+模糊处理)
def change_3(imgs,is_coarse=True):
    if (is_coarse): imgs = change_coarse(imgs)
    imgs=iaa.OneOf([iaa.GaussianBlur((1, 3.0)),  # blur images with a sigma between 0 and 3.0
            iaa.AverageBlur(k=(1, 3)), # blur image using local means with kernel sizes between 2 and 7
            iaa.MedianBlur(k=(1, 3))])(images=imgs)
    return imgs

# 4Coarse+HSV对比度变换
def change_4(imgs,is_coarse=True):
    if (is_coarse): imgs = change_coarse(imgs)
    imgs = iaa.Sequential([
        iaa.ContrastNormalization((0.75, 1.5), per_channel=True),  ####0.75-1.5随机数值为alpha，对图像进行对比度增强，该alpha应用于每个通道
        iaa.Multiply((0.5, 1.5), per_channel=0.5),####50%的图片像素值乘以0.5-1.5中间的数值,用以增加图片明亮度或改变颜色
        iaa.FrequencyNoiseAlpha(
            exponent=(-4, 0),
            first=iaa.Multiply((0.5, 1.5), per_channel=True),
            second=iaa.LinearContrast((0.5, 2.0))
        )
    ],random_order=True)(images=imgs)
    return imgs
#5+RGB颜色扰动
def change_5(imgs,is_coarse=True):
    if (is_coarse): imgs = change_coarse(imgs)
    imgs = iaa.OneOf([
        # 从离散的均匀范围中采样随机值[-50..50]，将其转换为角度表示形式，并将其添加到色相（即色空间中的H通道）中HSV。
        iaa.AddToHue((-50, 50)),
        # 通过随机值增加或减少色相和饱和度。
        # 增强器首先将图像转换为HSV色彩空间，然后向H和S通道添加随机值，然后转换回RGB。
        # 在色相和饱和度之间添加随机值（对于每个通道独立添加-100，100对于该通道内的所有像素均添加相同的值）。
        iaa.AddToHueAndSaturation((-100, 100), per_channel=True),
        # 将随机值添加到图像的饱和度。增强器首先将图像转换为HSV色彩空间，然后将随机值添加到S通道，然后转换回RGB。
        # 如果要同时更改色相和饱和度，建议使用AddToHueAndSaturation，否则图像将被两次转换为HSV并返RGB。
        # 从离散的均匀范围内采样随机值[-50..50]，并将其添加到饱和度，即添加到 色彩空间中的S通道HSV。
        iaa.AddToSaturation((-50, 50))
    ])(images=imgs)
    change = iaa.AddToBrightness((-30, 30))  # 每个图像转换成一个彩色空间与亮度相关的信道，提取该频道，之间加-30和30并转换回原始的色彩空间。
    imgs = change(images=imgs)
    return imgs
def get_new_flip_zoord(img,point,flip_data):
    is_lr, is_ud=flip_data
    h, w, _ = img.shape
    center_zoord=(w//2,h//2)
    return_x,return_y=(point[0],point[1])
    if(is_lr):
        return_x=2*center_zoord[0]-return_x
    if (is_ud):
        return_y = 2*center_zoord[1]-return_y
    return return_x,return_y
# point: 原图当中坐标点#     w: 原图宽#     h：原图高#     angle：旋转角度，逆时针方向
def get_new_rotate_zoord(img,point,angle):
    h,w,_=img.shape
    # print(w,h)
    # 计算原图当中的旋转中心，并将之设为原图当中的原点
    cx = w // 2
    cy = h // 2
    #先转到标准坐标系，左上为0，0点
    point_x,point_y=point[0],-point[1]
    cx,cy=cx,-cy
    offset_vector=(point_x-cx,point_y-cy)
    origin_angle = math.degrees(math.atan2(offset_vector[1], offset_vector[0]))
    new_angle = origin_angle - angle
    # 方便计算，把角度转为正负180度
    if new_angle > 180:
        new_angle -= 360
    # 计算当前点到新的原点，也就是旋转中心的距离
    lenth_p = math.sqrt(pow(offset_vector[0], 2) + pow(offset_vector[1], 2))
    # 使用角度与边长，计算到x轴和y轴的距离
    new_x = math.cos(math.radians(new_angle)) * lenth_p
    new_y = math.sin(math.radians(new_angle)) * lenth_p
    # 将坐标轴从旋转中心换回左上
    nx = int(new_x + cx)
    ny = int(new_y + cy)
    ny=-ny#上面是左上为标准坐标系下的坐标，取反是opencv图片下的坐标
    return nx,ny


# def test():
#     test_img2 = r"D:\工作资料\crowdcount_coach\data_increase\images\4_0.jpg"
#     img_src = cv2.imread(test_img2)
#     images = img_src[None, :]
#     sometimes = lambda aug: iaa.Sometimes(1, aug)
#     for i in range(5):
#         # change = iaa.Rotate((-45, 45))
#         change = iaa.OneOf([iaa.GaussianBlur((1, 2.0)),  # blur images with a sigma between 0 and 3.0
#                           iaa.AverageBlur(k=(2, 5)),  # blur image using local means with kernel sizes between 2 and 7
#                           iaa.MedianBlur(k=(3, 11))])(images=images)
#         change=iaa.MedianBlur(k=(1, 3))
#         change=iaa.AverageBlur(k=(1, 3))
#         # change=iaa.GaussianBlur((1, 3))
#         images_aug = change(images=images)
#         for i in range(images_aug.shape[0]):
#             img = images_aug[i]
#             # print(img.shape)(224, 224, 3)
#             cv2.imshow("", img)
#             cv2.waitKey(0)
#     # 增样种类：0原图 1Coarse+水平翻转和垂直翻转。2Coarse+旋转 3Coarse+彩色噪声(+模糊处理)4Coarse+HSV对比度变换+RGB颜色扰动
#     pass

def label_save(save_path,str):
    with open(save_path,"w") as f:
        f.write(str)

save_img_root_path="/Users/linzhihui/Documents/dataset/yolo_img/"
save_label_root_path="/Users/linzhihui/Documents/dataset/label/"

#第三类增样的数据使用各种均值模糊导致图片不清楚，这里删除后试下效果
# def del_some_pic():
#     del_img_root_path=r"/home/ubuntu/Public/yangc/train_data/crowdcount_coach/data_increase/images"
#     del_csv_root_path = r"/home/ubuntu/Public/yangc/train_data/crowdcount_coach/data_increase/labels"
#     for file in os.listdir(del_img_root_path):
#         img_path = os.path.join(del_img_root_path, file)
#         if(img_path.endswith("_3.jpg")):
#             file_del(img_path)
#     for file in os.listdir(del_csv_root_path):
#         csv_path = os.path.join(del_csv_root_path, file)
#         if(csv_path.endswith("_3.csv")):
#             file_del(csv_path)
#     pass
if __name__ == '__main__':
    # del_some_pic()
    # exit()
    if(not os.path.exists(save_img_root_path)):os.makedirs(save_img_root_path)
    if(not os.path.exists(save_label_root_path)):os.makedirs(save_label_root_path)
    save_img_idx = 0
    #生成增样数据，每一张图做change1--5，共变成6张图
    src_imgs_dirs=["/Users/linzhihui/Documents/dataset/val/"]
    # src_labels_dirs = [
    #     r"D:\工作资料\crowdcount_coach\labels",
    #     r"D:\工作资料\crowdcount_coach\add_byjson\labels"]
    src_jsons_dirs = ["/Users/linzhihui/Documents/dataset/label_json/"]
    for k, img_root_path in enumerate(src_imgs_dirs):
        for file in os.listdir(img_root_path):
            img_path = os.path.join(img_root_path, file)
            json_path=os.path.join(src_jsons_dirs[k],replace_suffix(file, ".json"))
            if (os.path.exists(json_path) ):
                save_img_idx += 1
                print(save_img_idx, img_path)
                # if(save_img_idx<=1050):continue
                img_array = cv_img_rgb(img_path)
                images = img_array[None, :]
                # images_aug1, data1 = change_1(images)
                # images_aug2, data2 = change_2(images)
                # images_aug3 = change_3(images)
                #images_aug4 = change_4(images)
                #images_aug5 = change_5(images)
                images_augs=[images]
                # np.set_printoptions(threshold=sys.maxsize)
                for i in range(len(images_augs)):
                    # if(i!=0):continue
                    save_img_full_name=os.path.join(save_img_root_path,str(file.split(".")[0])+".jpg".format(i))
                    save_label_full_name=os.path.join(save_label_root_path,str(file.split(".")[0])+".txt".format(i))
                    #save_img:
                    tmp_img_cv=images_augs[i][0]
                    if(i==0):
                        file_copy(img_path, save_img_full_name)
                    else:
                        img_save(tmp_img_cv,save_img_full_name)
                    #save_csv:
                    with open(json_path) as f:
                        json_dict = json.loads(f.read())
                        # print(json_dict)
                        data_array = json_dict["data"]
                        shape=tmp_img_cv.shape#544，960，3
                        mat_img_array = np.zeros(shape, dtype=np.float32)
                        str_label=""
                        data_draw=[]
                        points_draw=[]
                        for point in data_array:
                            new_zoord=point
                            #分情况讨论
                            #1如果高h在0--39 w设20，h设20
                            if(point[1]<20):
                                border_w, border_h = 1, 1
                            elif(point[1]<30):
                                border_w,border_h=2,2
                            
                            elif(point[1]<40):
                                border_w, border_h = 3, 3
                            elif(point[1]<80):
                                border_w, border_h = 5, 5
                            elif (point[1] < 120):
                                border_w, border_h = 7, 7
                            elif (point[1] < 200):
                                border_w, border_h = 15, 15
                            elif (point[1] < 300):
                                border_w, border_h = 60, 60
                            elif (point[1] < 400):
                                border_w, border_h = 85, 85
                            else:
                                border_w, border_h = 90, 90
                            offset_y=int(border_h*0.1)#中心点下移一点点
                            x1,y1=point[0]-border_w//2,point[1]-border_h//2+offset_y
                            x2,y2=point[0]+border_w//2,point[1]+border_h//2+offset_y
                            x1=min(max(0,x1),shape[1]-1)
                            y1=min(max(0,y1),shape[0]-1)
                            x2=min(max(0,x2),shape[1]-1)
                            y2=min(max(0,y2),shape[0]-1)
                            data_draw.append([(x1,y1),(x2,y2)])
                            # points_draw.append((new_zoord[0], new_zoord[1]))
                            h, w, _ = shape
                            center_x=(x1+x2)//2
                            center_y=(y1+y2)//2
                            width=x2-x1
                            height=y2-y1
                            str_label += "{0} {1} {2} {3} {4}\n". \
                                format(0, center_x / w, center_y / h, width / w,
                                       height / h)
                            try:
                                mat_img_array[new_zoord[1], new_zoord[0], :] = 1
                                # print(new_zoord)
                            except Exception as e:
                                print("idx err!!!!!!!!!!!!!!:",save_img_idx,img_path)
                        # mat_img_array_new = cv2.GaussianBlur(mat_img_array, gauss_ksize, gauss_sigma)
                        img_show=mat_img_array+tmp_img_cv/255.0
                        # print(str_label)
                        # for point0 in data_draw:
                        #     cv2.rectangle(img_show,point0[0],point0[1],color=[0,0,255],thickness=3)
                        # for point in points_draw:
                        #     cv2.circle(img_show,point,radius=5,color=(255,0,0))
                        # cv2.imshow("2", img_show)
                        # cv2.waitKey(0)
                        # exit()s

                        label_save(save_label_full_name, str_label)
                        # save_array_as_csv(mat_img_array_new[:, :, 0], save_label_full_name)
                        # exit()
                    pass
    pass