from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess

from tensorflow.keras.models import Model
import numpy as np


class PreTrainedModelCNNBased(object):
    def __init__(self):
        print("This is toolkit of cnn model")

    @staticmethod
    def vgg16(weights='imagenet', include_top=False):
        model = VGG16(weights=weights, include_top=include_top)
        print(model.summary())
        return model

    @staticmethod
    def vgg19(weights='imagenet', include_top=False):
        model = VGG19(weights=weights, include_top=include_top)
        print(model.summary())
        return model

    @staticmethod
    def inceptionV3(weights='imagenet', include_top=False):
        model = InceptionV3(weights=weights, include_top=include_top)
        print(model.summary())
        return model

    @staticmethod
    def xception(weights='imagenet', include_top=False):
        model = Xception(weights=weights, include_top=False)
        print(model.summary())
        return model

    @staticmethod
    def extract_feature(model, image, layer_name=None, preprocess_type='vgg16'):
        assert len(image.shape) == 3
        x = np.expand_dims(image, axis=0).astype(dtype='float32')
        if preprocess_type == 'vgg16':
            x = vgg16_preprocess(x)
        if preprocess_type == 'vgg19':
            x = vgg19_preprocess(x)

        if layer_name is None:
            return model.predict(x)
        else:
            outputs = []
            for l in layer_name:
                outputs.append(model.get_layer(l).output)
            model_new = Model(inputs=model.input, outputs=outputs)
            feats = model_new.predict(x)
            return dict(zip(layer_name, feats))


if __name__ == '__main__':
    import utils.my_utils as my_io
    pretrained_models = PreTrainedModelCNNBased()
    model = pretrained_models.vgg19()
    test_img = '../test_data/me.jpeg'
    img = image.load_img(test_img, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    feat = pretrained_models.extract_feature(model, img, layer_name=('block4_conv1', 'block5_conv1'), preprocess_type='vgg19')
    print(feat)
