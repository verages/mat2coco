import json
import os
import cv2
import numpy
import numpy as np

from tools import replace_suffix, cv_img_rgb, get_point_size_img, save_array_as_csv, get_json_data, img_save



def gen_json_points(points_array,save_img_idx,save_jsons_path="/Users/linzhihui/Documents/dataset/label_json/"):
    save_full_path=os.path.join(save_jsons_path,str(save_img_idx)+".json")
    if(type(points_array) is numpy.ndarray):
        points_array=points_array.tolist()
    dictObj = {
        'data': points_array,
    }
    jsObj = json.dumps(dictObj)
    fileObject = open(save_full_path, 'w')
    fileObject.write(jsObj)
    fileObject.close()


save_imgs_path=r"D:\工作资料\crowdcount_coach\add_byjson\images"
save_labels_path=r"D:\工作资料\crowdcount_coach\add_byjson\labels"


img_size_h_w = (544,960)
gauss_ksize = (7 ,13)
gauss_sigma = 3#(9, 5)
if __name__ == '__main__':
    if(not os.path.exists(save_imgs_path)):
        os.makedirs(save_imgs_path)
    if (not os.path.exists(save_labels_path)):
        os.makedirs(save_labels_path)
    img_root_path=r"D:\工作资料\crowdcount_coach_need_mark"
    json_root_path=r"D:\工作资料\crowdcount_coach_need_mark"
    cur_save_pic_idx=0
    for file in os.listdir(img_root_path):
        img_path = os.path.join(img_root_path, file)
        if not (os.path.isfile(img_path) and file.endswith(".jpg")):continue
        json_path = os.path.join(json_root_path, replace_suffix(file, ".json"))
        cur_save_pic_idx += 1
        print(cur_save_pic_idx)
        if (os.path.exists(json_path)):
            points_arr = get_json_data(json_path)
            img_array = cv_img_rgb(img_path)
            img_array_new, ratio, (offset_y, offset_x) = get_point_size_img(img_array, img_size_h_w)
            mat_img_array = np.zeros(img_array_new.shape, dtype=np.float32)
            h_new, w_new, _ = img_array_new.shape
            points=[]
            for point in points_arr:
                # cv2.circle(img_array, (int(points_arr[i][0]), int(points_arr[i][1])), point_size, point_color, thickness)
                old_zoord_x_y = [int(point[0]), int(point[1])]
                # 换算，原坐标换算到新坐标
                # 1先换算缩放
                new_old_zoord_x_y = [int(old_zoord_x_y[0] * ratio), int(old_zoord_x_y[1] * ratio)]
                # 2在换算偏移
                new_old_zoord_x_y = [new_old_zoord_x_y[0] + offset_x, new_old_zoord_x_y[1] + offset_y]
                if (new_old_zoord_x_y[1] >= h_new): new_old_zoord_x_y[1] = h_new - 1
                if (new_old_zoord_x_y[0] >= w_new): new_old_zoord_x_y[0] = w_new - 1
                mat_img_array[new_old_zoord_x_y[1], new_old_zoord_x_y[0], :] = 1
                points.append(new_old_zoord_x_y)
            # mat_img_array_new = cv2.GaussianBlur(mat_img_array, gauss_ksize, gauss_sigma)
            # cv2.imshow("1",img_array_new)
            # mat_img_array_new=mat_img_array_new*255
            # cv2.imshow("2", mat_img_array_new+img_array_new/255)
            # cv2.waitKey(0)
            # save_img_full_path = os.path.join(save_imgs_path, str(cur_save_pic_idx) + ".jpg")
            # img_save(img_array_new, save_img_full_path)
            # save_csv_full_path = os.path.join(save_labels_path, str(cur_save_pic_idx) + ".csv")
            # save_array_as_csv(mat_img_array_new[:, :, 0], save_csv_full_path)
            gen_json_points(points, cur_save_pic_idx,save_jsons_path=r"D:\工作资料\crowdcount_coach\add_byjson\jsons")
            pass
        else:
            # img_array = cv_img_rgb(img_path)
            # img_array_new, ratio, (offset_y, offset_x) = get_point_size_img(img_array, img_size_h_w)
            # #如果没有表示这个图没有人头，直接标签就是0的csv
            # mat_img_array_new = np.zeros(img_array_new.shape, dtype=np.float32)
            # save_img_full_path = os.path.join(save_imgs_path, str(cur_save_pic_idx) + ".jpg")
            # img_save(img_array_new, save_img_full_path)
            # save_csv_full_path = os.path.join(save_labels_path, str(cur_save_pic_idx) + ".csv")
            # save_array_as_csv(mat_img_array_new[:, :, 0], save_csv_full_path)
            points=[]
            gen_json_points(points, cur_save_pic_idx,save_jsons_path=r"D:\工作资料\crowdcount_coach\add_byjson\jsons")
            pass
