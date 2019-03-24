from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetLarge
from keras.layers import Input

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

model01 = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
model02 = Xception(input_tensor=input_tensor, weights='imagenet', include_top=True)
model03 = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=True)
model04 = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=True)
model05 = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=True)
model09 = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet', include_top=True)
model10 = MobileNet(input_tensor=input_tensor, weights='imagenet', include_top=True)
model12 = DenseNet201(input_tensor=input_tensor, weights='imagenet', include_top=True)
model13 = NASNetLarge(input_tensor=input_tensor, weights='imagenet', include_top=True)
