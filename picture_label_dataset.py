import tensorflow as tf
from  tensorflow import keras
import matplotlib.pyplot as plt
import IPython.display as display
from keras.applications.inception_v3 import InceptionV3

tf.enable_eager_execution()
print(tf.VERSION)
AUTOTUNE = tf.data.experimental.AUTOTUNE#supported in 1.13.0

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [224, 224])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


pictures_labels=open("frames_target.txt", "r")
all_image_paths=[' '.join(x.split()[:-1]) for x in pictures_labels]
pictures_labels=open("frames_target.txt", "r")
all_image_labels=[x.split()[-1] for x in pictures_labels]

label_names = sorted(set(all_image_labels))
print(label_names)
label_to_index = dict((name, index) for index,name in enumerate(label_names))
print(label_to_index)
all_image_labels=[label_to_index[l] for l in all_image_labels]

image_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(image_path))
plt.grid(False)
plt.xlabel(image_path)
plt.title(label)
# plt.show()
#----------------end of data examination---------------
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
for n,image in enumerate(image_ds.take(4)):
    plt.subplot(2,2,n+1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(all_image_paths[n])
# plt.show()

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))



image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

print('image shape: ', image_label_ds.output_shapes[0])
print('label shape: ', image_label_ds.output_shapes[1])
print('types: ', image_label_ds.output_types)
print()
print(image_label_ds)


BATCH_SIZE = 32

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_label_ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=len(all_image_labels))
)
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds


# this could also be the output a different Keras model or layer
# input_tensor = keras.layers.Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

# model01 = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
# model01.trainable=False
# model01.compile(optimizer=tf.train.AdamOptimizer(), 
#               loss=tf.keras.losses.sparse_categorical_crossentropy,
#               metrics=["accuracy"])
# model01.predict(ds)

#https://www.tensorflow.org/tutorials/load_data/images