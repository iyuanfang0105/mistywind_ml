import tensorflow as tf
import numpy as np
import functools
import time
import matplotlib.pyplot as plt

from vgg19 import VGG19
import utils
import transform

STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
CONTENT_LAYER = ['relu4_2']

image_max_size = 256
batch_size = 5


vgg19_weights = '../vgg_weights/imagenet-vgg-verydeep-19.mat'
vgg_inputs = tf.placeholder(dtype=tf.float32, shape=(None, image_max_size, image_max_size, 3), name='inputs')
vgg19_model = VGG19(vgg19_weights)
vgg19_model.build_model(vgg_inputs, verbose=True)


content_img_path = '../../test_data/me.jpeg'
style_img_path = '../style_images/shipwreck.jpg'
content_img = utils.read_image(content_img_path, preprocess_flag=True, max_size=image_max_size)
content_img = np.concatenate((content_img, content_img), axis=0)
style_img = utils.read_image(style_img_path, preprocess_flag=True, max_size=image_max_size)
print(style_img.shape)

# get sytle features
with tf.Session() as sess:
    # style_features = {}
    # for l in STYLE_LAYERS:
        # style_features[l] = sess.run(vgg19_model.net[l], feed_dict={inputs: style_img})
    style_features = vgg19_model.get_feature(sess, STYLE_LAYERS, style_img, gram=True)


# get content and predict features, and define loss
with tf.Session() as sess:
    transform_inputs = tf.placeholder(tf.float32, shape=(batch_size, image_max_size, image_max_size, 3), name='transform_inputs')
    content_features = {}
    for l in CONTENT_LAYER:
        content_features[l] = vgg19_model.net[l]

    pred_image = transform.net(transform_inputs)

    pred_features = {}
    for l in CONTENT_LAYER+STYLE_LAYERS:
        pred_features[l] = vgg19_model.net[l]

    content_loss = tf.Variable(0.0)
    for l in CONTENT_LAYER:
        assert utils.tensor_size(pred_features[l]) == utils.tensor_size(content_features[l])
        content_loss += (2 * tf.nn.l2_loss(pred_features[l] - content_features[l])) / (utils.tensor_size(pred_features[l]) * batch_size)

    style_losses = []
    for l in STYLE_LAYERS:
        pred_layer_gram = utils.gram_matrix(pred_features[l], -1, pred_features[l].shape[-1])
        style_losses.append(2 * tf.nn.l2_loss(pred_layer_gram - style_features[l]) / style_features[l].size)
    style_loss = functools.reduce(tf.add, style_losses) / batch_size

    loss = content_loss + style_loss


with tf.Session() as sess:
    train_step = tf.train.AdamOptimizer(learning_rate=1e3).minimize(loss)
    sess.run(tf.global_variables_initializer())

    content_image_files = utils.get_files('test_image')

    for epoch in range(1000):
        num_examples = len(content_image_files)
        iters = 0
        while iters * batch_size < num_examples:
            st = time.time()
            curr = iters * batch_size
            step = curr + batch_size
            X_batch = np.zeros((batch_size, image_max_size, image_max_size, 3), dtype=np.float32)
            for j, image_path in enumerate(content_image_files[curr:step]):
                X_batch[j] = utils.read_image(image_path, preprocess_flag=True, max_size=image_max_size)

            iters += 1
            assert X_batch.shape[0] == batch_size

            feed_dict = {transform_inputs: X_batch}

            train_step.run(feed_dict=feed_dict)
            dt = time.time() - st

            print("epoch_%d-batch_%d, time collapse: %s" % (epoch, iters, dt))

        if epoch > 0 and epoch % 20 == 0:
            test_feed_dict = {transform_inputs: X_batch}
            pd_img = sess.run(pred_image, feed_dict=test_feed_dict)
            plt.imshow(pd_img[0])
            plt.pause(0.05)