import os
import tensorflow as tf
import time
import cv2

import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util

import transform
import utils


pretrained_style_file = 'pretrained_styles/rain_princess.ckpt'
batch_size = 1
image_max_size = 400
content_img_path = '../../test_data/me.jpeg'

content_img = utils.read_image(content_img_path, preprocess_flag=True, max_size=image_max_size)


g = tf.Graph()
soft_config = tf.ConfigProto(allow_soft_placement=True)
with g.as_default(), tf.Session(config=soft_config) as sess:
    img_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_max_size, image_max_size, 3),
                                             name='img_placeholder')
    preds = transform.net(img_placeholder)
    saver = tf.train.Saver()

    saver.restore(sess, pretrained_style_file)

    # output_node_names = ['Tanh']
    # output_graph_def = tf.graph_util.convert_variables_to_constants(
    #     sess,  # The session is used to retrieve the weights
    #     tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
    #     output_node_names  # The output node names are used to select the usefull nodes
    # )
    # with tf.gfile.GFile('test/aa.pb', "wb") as f:
    #     f.write(output_graph_def.SerializeToString())

    st = time.time()
    pred_v = sess.run(preds, feed_dict={img_placeholder: content_img})
    pred_v = utils.postprocess(pred_v)
    dt = time.time() - st
    print(dt)
    cv2.imwrite('test_image/me.output.jpg', pred_v)
