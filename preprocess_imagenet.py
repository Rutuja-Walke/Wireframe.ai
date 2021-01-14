from keras.applications import VGG19
from keras.layers import Dense,GlobalAveragePooling2D,Flatten, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

import pandas as pd
import numpy as np

import shutil
import os
base_model=VGG19(weights='imagenet',include_top=False)
x1=base_model.output

model=Model(inputs=base_model.input,outputs=x1)
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
print(len(model.layers))

def get_preprocessed_img(img_path, image_size):
    import cv2
    img = cv2.imread(img_path)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype('float32')
    img /= 255
    return img

input_path="./all_data"
output_path="./resources/data"

for f in os.listdir(input_path):
    if f.find(".png") != -1:
        img = get_preprocessed_img("{}/{}".format(input_path, f), 224)
        file_name = f[:f.find(".png")]
        print(file_name)
        img=np.reshape(np.array(img),(1,224,224,3))
        print(img.shape)
        predict=model.predict(img)
        np.savez_compressed("{}/{}".format(output_path, file_name), features=predict)
        retrieve = np.load("{}/{}.npz".format(output_path, file_name))["features"]

        print(len(predict))
        assert np.array_equal(predict, retrieve)

        shutil.copyfile("{}/{}.gui".format(input_path, file_name), "{}/{}.gui".format(output_path, file_name))
