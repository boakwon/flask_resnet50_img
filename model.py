import base64

import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image


# class ResNet50Model():
#     def model_load(self):
#         model = ResNet50(weights='imagenet')
#         print('Model loaded. Check http://127.0.0.1:5000/ or http://localhost:5000/')
#         return model
#
#     def model_predict(self,img_path, model):
#         img = image.load_img(img_path, target_size=(224, 224))
#
#         # Preprocessing the image
#         x = image.img_to_array(img)
#         # x = np.true_divide(x, 255)
#         x = np.expand_dims(x, axis=0)
#
#         # Be careful how your trained model deals with the input
#         # otherwise, it won't make correct prediction!
#         x = preprocess_input(x, mode='caffe')
#
#         preds = model.predict(x)
#         return preds


img_path = 'uploads/car.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print(x)


def img_to_base64(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    encode_image = cv2.imencode(".jpg", img_array)[1]
    byte_data = encode_image.tobytes()
    base64_str = base64.b64encode(byte_data).decode("ascii")
    return base64_str

base64_str = img_to_base64(img_path)
print(base64_str)