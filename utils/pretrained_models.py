from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np


def vgg16(weights='imagenet', include_top=False):
    model = VGG16(weights=weights, include_top=include_top)
    print(model.summary())
    return model


def vgg19(weights='imagenet', include_top=False):
    model = VGG19(weights=weights, include_top=include_top)
    print(model.summary())
    return model


def inceptionV3(weights='imagenet', include_top=False):
    model = InceptionV3(weights=weights, include_top=include_top)
    print(model.summary())
    return model


def xception(weights='imagenet', include_top=False):
    model = Xception(weights=weights, include_top=False)
    print(model.summary())
    return model


def extract_feature(model, image, layer_name=None):
    assert len(image.shape) == 3
    x = np.expand_dims(image, axis=0)
    x = preprocess_input(x)

    if layer_name is None:
        return model.predict(x)
    else:
        model_new = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        return model_new.predict(x)


if __name__ == '__main__':
    import utils.my_io as my_io
    model = xception()
    test_img = '/Users/wind/WORK/code/tf_learning/test_data/WechatIMG7.jpeg'
    img = my_io.load_image(test_img, target_size=(224, 224))
    feat = extract_feature(model, img)
    print(feat)
