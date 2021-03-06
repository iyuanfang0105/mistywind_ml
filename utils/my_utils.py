# coding=utf-8

import sys
import os
import urllib
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def get_file_list(folder_name, recursive=False):
    folder_info = os.listdir(folder_name)
    file_list = []
    for f in folder_info:
        f_path = os.path.join(folder_name, f)
        if os.path.isdir(f_path) and recursive:
            file_list.extend(get_file_list(f_path, recursive=recursive))
        else:
            file_list.append(f_path)
    return file_list


def is_image(file_path, use_extention=True, image_type=None, use_pil=False):
    if image_type is None:
        image_type = ['jpg', 'jpeg', 'bmp', 'png', 'gif']

    result = None
    if use_extention:
        result = any(file_path.endswith(ext) for ext in image_type)
    if use_pil:
        try:
            Image.open(file_path)
        except IOError:
            print('PIL format error')
        result = True
    return result


def read_image(img_path, show=False, to_rgb=True, target_size=None):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if target_size:
        img = cv2.resize(img, target_size)

    # img = image.load_img(img_path, target_size=target_size)
    # x = image.img_to_array(img)

    if to_rgb:
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        if show:
            img_name = os.path.basename(img_path)
            show_image(img, img_name)

    return img


def show_image(image, image_name=None):
    plt.imshow(image)
    if image_name:
        plt.title(image_name)
    plt.show()


def read_text_file(file_name):
    file_data = []
    with open(file_name, 'r') as fi:
        for line in fi:
            file_data.append(line.strip().split(" "))
    return file_data


def read_traslation_corpus(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").split('\n')



if __name__ == '__main__':
    # test_folder_name = '/Users/wind/WORK/public_data_set/quick_draw/raw'
    # a = get_file_list(test_folder_name, recursive=False)
    # print(a)
    # print(len(a))

    # test_img = '/Users/wind/WORK/code/tf_learning/test_data/WechatIMG7.jpeg'
    # img = load_image(test_img)
    # print()

    test_text = '/Users/wind/WORK/public_data_set/text8/text8'
    data = read_text_file(test_text)
    print(len(data))
    print(data[0][:7])