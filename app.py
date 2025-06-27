import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import pickle

import os
model=ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainable=False
#APNE MODEL MEI HAMNE EK LAYER ADD KARDI LAST VALI HATAKE HAMARI
model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
# print(model.summary())

def extract_features(path,model):
    i=image.load_img(path,target_size=(224,224))
    array_of_image=image.img_to_array(i)
    expand_image_array=np.expand_dims(array_of_image,axis=0)
    preprocess_image=preprocess_input(expand_image_array)
    result=model.predict(preprocess_image).flatten()
    normalised_result=result/norm(result)
    return normalised_result
filenames=[]
for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))
# print(len(filenames))
# print(filenames[0:5])

feature_list=[]
for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))
# print(np.array(feature_list).shape)
pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))