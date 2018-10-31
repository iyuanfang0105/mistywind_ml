import os
import sys
import numpy as np
import cv2
import errno
import functools
import tensorflow as tf


def check_image(img, path):
    if img is None:
        raise OSError(errno.ENOENT, "No such file", path)


def preprocess(img):
    imgpre = np.copy(img)
    # bgr to rgb
    imgpre = imgpre[..., ::-1]
    # shape (h, w, d) to (1, h, w, d)
    imgpre = imgpre[np.newaxis, :, :, :]
    imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    return imgpre


def postprocess(img):
    imgpost = np.copy(img)
    imgpost += np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    # shape (1, h, w, d) to (h, w, d)
    imgpost = imgpost[0]
    imgpost = np.clip(imgpost, 0, 255).astype('uint8')
    # rgb to bgr
    imgpost = imgpost[..., ::-1]
    return imgpost


def read_image(path, preprocess_flag=False, max_size=500):
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    img = img.astype(np.float32)
    h, w, d = img.shape

    # # resize if > max size
    # if max_size:
    #     if h > w and h > max_size:
    #         w = (float(max_size) / float(h)) * w
    #         img = cv2.resize(img, dsize=(int(w), max_size), interpolation=cv2.INTER_AREA)
    #     if w > max_size:
    #         h = (float(max_size) / float(w)) * h
    #         img = cv2.resize(img, dsize=(max_size, int(h)), interpolation=cv2.INTER_AREA)
    if max_size:
        img = cv2.resize(img, dsize=(max_size, max_size), interpolation=cv2.INTER_AREA)

    if preprocess_flag:
        img = preprocess(img)
    return img


def cv2_show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]


def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break
    return files


def tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G

if __name__ == '__main__':
    img_path = 'test_image'
    files = get_files(img_path)
    img = read_image(img_path, preprocess_flag=True)
    print()