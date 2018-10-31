import tensorflow as tf
import scipy.io
import numpy as np
import utils


class VGG19(object):
    def __init__(self, weights_file):
        self.weights_file = weights_file
        self.net = None

    @staticmethod
    def conv_layer(layer_name, layer_input, W, verbose=False):
        conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding='SAME')
        if verbose:
            print('--{} | shape={} | weights_shape={}'.format(layer_name, conv.get_shape(), W.get_shape()))
        return conv

    @staticmethod
    def relu_layer(layer_name, layer_input, b, verbose=False):
        relu = tf.nn.relu(layer_input + b)
        if verbose:
            print('--{} | shape={} | bias_shape={}'.format(layer_name, relu.get_shape(), b.get_shape()))
        return relu

    @staticmethod
    def pool_layer(layer_name, layer_input, pooling_type='max', verbose=False):
        if pooling_type == 'avg':
            pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        elif pooling_type == 'max':
            pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        if verbose:
            print('--{} | shape={}'.format(layer_name, pool.get_shape()))
        return pool

    @staticmethod
    def get_weights(vgg_layers, i):
        weights = vgg_layers[i][0][0][2][0][0]
        W = tf.constant(weights)
        return W

    @staticmethod
    def get_bias(vgg_layers, i):
        bias = vgg_layers[i][0][0][2][0][1]
        b = tf.constant(np.reshape(bias, (bias.size)))
        return b

    def build_model(self, input, verbose=False):
        if verbose:
            print('\nBUILDING VGG-19 NETWORK')
        net = {}
        # _, h, w, d = input_img.shape

        if verbose:
            print('loading model weights...')

        vgg_rawnet = scipy.io.loadmat(self.weights_file)
        vgg_layers = vgg_rawnet['layers'][0]
        if verbose:
            print('constructing layers...')
        # net['input'] = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))
        # net['input'] = tf.placeholder(tf.float32, shape=(_, h, w, d), name='input')
        # net['input'] = tf.placeholder(tf.float32, name='input')

        if verbose:
            print('LAYER GROUP 1')
        net['input'] = input
        net['conv1_1'] = VGG19.conv_layer('conv1_1', net['input'], W=VGG19.get_weights(vgg_layers, 0), verbose=verbose)
        net['relu1_1'] = VGG19.relu_layer('relu1_1', net['conv1_1'], b=VGG19.get_bias(vgg_layers, 0), verbose=verbose)

        net['conv1_2'] = VGG19.conv_layer('conv1_2', net['relu1_1'], W=VGG19.get_weights(vgg_layers, 2), verbose=verbose)
        net['relu1_2'] = VGG19.relu_layer('relu1_2', net['conv1_2'], b=VGG19.get_bias(vgg_layers, 2), verbose=verbose)

        net['pool1'] = VGG19.pool_layer('pool1', net['relu1_2'], verbose=verbose)

        if verbose:
            print('LAYER GROUP 2')
        net['conv2_1'] = VGG19.conv_layer('conv2_1', net['pool1'], W=VGG19.get_weights(vgg_layers, 5), verbose=verbose)
        net['relu2_1'] = VGG19.relu_layer('relu2_1', net['conv2_1'], b=VGG19.get_bias(vgg_layers, 5), verbose=verbose)

        net['conv2_2'] = VGG19.conv_layer('conv2_2', net['relu2_1'], W=VGG19.get_weights(vgg_layers, 7), verbose=verbose)
        net['relu2_2'] = VGG19.relu_layer('relu2_2', net['conv2_2'], b=VGG19.get_bias(vgg_layers, 7), verbose=verbose)

        net['pool2'] = VGG19.pool_layer('pool2', net['relu2_2'], verbose=verbose)

        if verbose:
            print('LAYER GROUP 3')
        net['conv3_1'] = VGG19.conv_layer('conv3_1', net['pool2'], W=VGG19.get_weights(vgg_layers, 10), verbose=verbose)
        net['relu3_1'] = VGG19.relu_layer('relu3_1', net['conv3_1'], b=VGG19.get_bias(vgg_layers, 10), verbose=verbose)

        net['conv3_2'] = VGG19.conv_layer('conv3_2', net['relu3_1'], W=VGG19.get_weights(vgg_layers, 12), verbose=verbose)
        net['relu3_2'] = VGG19.relu_layer('relu3_2', net['conv3_2'], b=VGG19.get_bias(vgg_layers, 12), verbose=verbose)

        net['conv3_3'] = VGG19.conv_layer('conv3_3', net['relu3_2'], W=VGG19.get_weights(vgg_layers, 14), verbose=verbose)
        net['relu3_3'] = VGG19.relu_layer('relu3_3', net['conv3_3'], b=VGG19.get_bias(vgg_layers, 14), verbose=verbose)

        net['conv3_4'] = VGG19.conv_layer('conv3_4', net['relu3_3'], W=VGG19.get_weights(vgg_layers, 16), verbose=verbose)
        net['relu3_4'] = VGG19.relu_layer('relu3_4', net['conv3_4'], b=VGG19.get_bias(vgg_layers, 16), verbose=verbose)

        net['pool3'] = VGG19.pool_layer('pool3', net['relu3_4'], verbose=verbose)

        if verbose:
            print('LAYER GROUP 4')
        net['conv4_1'] = VGG19.conv_layer('conv4_1', net['pool3'], W=VGG19.get_weights(vgg_layers, 19), verbose=verbose)
        net['relu4_1'] = VGG19.relu_layer('relu4_1', net['conv4_1'], b=VGG19.get_bias(vgg_layers, 19), verbose=verbose)

        net['conv4_2'] = VGG19.conv_layer('conv4_2', net['relu4_1'], W=VGG19.get_weights(vgg_layers, 21), verbose=verbose)
        net['relu4_2'] = VGG19.relu_layer('relu4_2', net['conv4_2'], b=VGG19.get_bias(vgg_layers, 21), verbose=verbose)

        net['conv4_3'] = VGG19.conv_layer('conv4_3', net['relu4_2'], W=VGG19.get_weights(vgg_layers, 23), verbose=verbose)
        net['relu4_3'] = VGG19.relu_layer('relu4_3', net['conv4_3'], b=VGG19.get_bias(vgg_layers, 23), verbose=verbose)

        net['conv4_4'] = VGG19.conv_layer('conv4_4', net['relu4_3'], W=VGG19.get_weights(vgg_layers, 25), verbose=verbose)
        net['relu4_4'] = VGG19.relu_layer('relu4_4', net['conv4_4'], b=VGG19.get_bias(vgg_layers, 25), verbose=verbose)

        net['pool4'] = VGG19.pool_layer('pool4', net['relu4_4'], verbose=verbose)

        if verbose:
            print('LAYER GROUP 5')
        net['conv5_1'] = VGG19.conv_layer('conv5_1', net['pool4'], W=VGG19.get_weights(vgg_layers, 28), verbose=verbose)
        net['relu5_1'] = VGG19.relu_layer('relu5_1', net['conv5_1'], b=VGG19.get_bias(vgg_layers, 28), verbose=verbose)

        net['conv5_2'] = VGG19.conv_layer('conv5_2', net['relu5_1'], W=VGG19.get_weights(vgg_layers, 30), verbose=verbose)
        net['relu5_2'] = VGG19.relu_layer('relu5_2', net['conv5_2'], b=VGG19.get_bias(vgg_layers, 30), verbose=verbose)

        net['conv5_3'] = VGG19.conv_layer('conv5_3', net['relu5_2'], W=VGG19.get_weights(vgg_layers, 32), verbose=verbose)
        net['relu5_3'] = VGG19.relu_layer('relu5_3', net['conv5_3'], b=VGG19.get_bias(vgg_layers, 32), verbose=verbose)

        net['conv5_4'] = VGG19.conv_layer('conv5_4', net['relu5_3'], W=VGG19.get_weights(vgg_layers, 34), verbose=verbose)
        net['relu5_4'] = VGG19.relu_layer('relu5_4', net['conv5_4'], b=VGG19.get_bias(vgg_layers, 34), verbose=verbose)

        net['pool5'] = VGG19.pool_layer('pool5', net['relu5_4'], verbose=verbose)

        self.net = net

    def get_feature(self, sess, layers, image, gram=False):
        assert self.net is not None, "ERROR!!!! Please build model first"
        features = {}
        for l in layers:
            feat = sess.run(self.net[l], feed_dict={self.net['input']: image})
            if gram:
                feat = utils.gram_matrix(feat, -1, feat.shape[3])
            features[l] = feat.eval()
        return features




if __name__ == '__main__':
    import utils

    test_img = '../../test_data/me.jpeg'
    img = utils.read_image(test_img, preprocess_flag=True)

    vgg19_weights = '../vgg_weights/imagenet-vgg-verydeep-19.mat'
    vgg19_model = VGG19(vgg19_weights)
    content_image_tf = tf.placeholder(dtype=tf.float32, shape=(4, 256, 256, 3), name='placeholder_content_images')
    vgg19_model.build_model(content_image_tf, verbose=True)
    # vgg19_model.build_model(img, vgg19_weights, verbose=True)
    #
    # layers = ['conv4_2']
    # with tf.Session() as sess:
    #     features = vgg19_model.get_feature(sess, layers, img)

    print()