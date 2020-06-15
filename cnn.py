#%%
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow.keras.callbacks import EarlyStopping, Callback
import warnings
import os

print(tf.test.is_gpu_available())
print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMG_SHAPE = 224
class_names = os.listdir('imagescopy/train/')
print(class_names)

train_dir = 'imagescopy/train/'
test_dir = 'imagescopy/test/'

#%%
batch_size = 64
epochs = 200

# %%
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    preprocessing_function=preprocess_input,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    #horizontal_flip=True,
                    zoom_range=0.5
                    )


train_data_gen = image_gen_train.flow_from_directory(
                                                batch_size=batch_size,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE,IMG_SHAPE),
                                                class_mode='categorical',
                                                )
image_gen_val = ImageDataGenerator(
                                    rescale=1./255,
                                    preprocessing_function=preprocess_input
                                    )

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=test_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='categorical',
                                                 shuffle=False)

#%%
weight = 'D:/Project/ResNet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
weight_linux = '/media/benayac/Data/Project/ResNet/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
def get_base_model():
    model = VGG19(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-2]._outbound_nodes= []
    x = Conv2D(256, kernel_size=(2,2),strides=2)(model.output)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    x = Conv2D(128, kernel_size=(2,2),strides=1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(len(class_names), activation='softmax')(x)
    model=Model(model.input,x)

    for layer in model.layers[:22]:
        layer.trainable = False

    return model

model = get_base_model()

#%%
class EarlyStoppingValAcc(Callback):
    def __init__(self, monitor=['val_accuracy', 'accuracy'], value=[0.90, 0.90], verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') > self.value[0] and logs.get('accuracy') > self.value[1]:
            print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

#%%
callback = EarlyStoppingValAcc(monitor=['val_accuracy', 'accuracy'], value=[0.94, 0.95], verbose=1)
optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
#%%
model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n/batch_size)),
    validation_data=val_data_gen,
    epochs=epochs,
    validation_steps=int(np.ceil(val_data_gen.n/batch_size)),
    shuffle=True,
    callbacks=[callback]
)

# %%
import numpy as np
from PIL import Image
import cv2
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.python.keras.applications.vgg19 import preprocess_input

class_names = os.listdir('imagescopy/train/')
print(class_names)
folder = 'candi_brahu'

url = 'https://www.balitoursclub.net/wp-content/uploads/2016/06/Objek-wisata-Gunung-Kawi-Bali-300x183.jpg'
response = requests.get(url)
img = img_to_array(load_img(BytesIO(response.content)).resize((224, 224)))
img = preprocess_input(img)
img = img/255.
img = np.expand_dims(img, axis=0)
result = model.predict(img)
prediction = class_names[np.argmax(result[0], axis=-1)]
print(max(result[0])*100)
print(prediction)

# imgs = [img_to_array(load_img(f'imagescopy/test/{folder}/{img}').resize((224, 224))) for img in os.listdir(f'imagescopy/test/{folder}')]
# print(len(imgs))
# score = 0

# for img in imgs:
#     img = preprocess_input(img)
#     img = img/255.
#     img = np.expand_dims(img, axis=0)
#     result = model.predict(img)
#     prediction = class_names[np.argmax(result[0], axis=-1)]
#     if(prediction == folder):
#         score+=1
# print(f'Accuracy = {score/len(imgs)}')

# %%
model.save('my_model_v4')


# %%
model = tf.keras.models.load_model('my_model_v4')

# %%
